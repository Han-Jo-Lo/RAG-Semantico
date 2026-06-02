import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import  SystemMessage,RemoveMessage,AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

from config import DATABASE_DIRECTORY_NAME, get_llm
from vector_store import VectorStoreManager

from config import get_redis_saver


llm = get_llm()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente de soporte técnico experto. Tu objetivo es ayudar "
    "a los usuarios basándote exclusivamente en el contexto proporcionado.\n"
    "REGLAS CRÍTICAS:\n"
    "1. Usa el CONTEXTO_RECUPERADO para responder.\n"
    "2. Si la respuesta no está en el contexto, di: 'Lo siento, no tengo esa información'.\n"
    "3. Responde siempre en español de forma profesional."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_context: str
    summary:str
    valid_answer:bool
    has_context:bool
    standalone_question:str

def build_app_graph(vector_store: VectorStoreManager):

    def reformulate_node(state: State):
        messages = state["messages"]
        summary = state.get("summary", "")

        
        if len(messages) <= 1:
            return {"standalone_question": messages[-1].content}

        summary_context = f"Resumen previo:\n{summary}\n\n" if summary else ""


        recent_messages = messages[-6:]

        prompt = [
            SystemMessage(content=(
                f"{summary_context}"
                "Dado el historial de conversación, reformula la ÚLTIMA pregunta "
                "del usuario en una pregunta autosuficiente y clara, sin pronombres "
                "ambiguos. Responde SOLO con la pregunta reformulada, sin explicaciones."
            ))
        ] + recent_messages

        result = llm.invoke(prompt)
        return {"standalone_question": result.content}

    def context_node(state:State):
        pregunta=state['standalone_question']
        result=vector_store.retrieve_with_score(pregunta)

        relevant_docs=[doc for doc,score in result if score<0.67]
        context_str = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)


        has_context=len(relevant_docs)>0
        print('--------'+str(has_context))
        return {
            "retrieved_context": context_str,
            'has_context':has_context
            }


    def should_continue(state:State):
        if state['has_context']:
            return 'chatbot'
        else:
            return 'no_answer'

    def chatbot_node(state: State):
        context = state["retrieved_context"]
        #print ('====='+context)
        messages = state["messages"]
        summary=state.get('summary','')
        summary_context=f'Resumen de la conversacion previa: {summary}\n' if summary else ''

        prompt_msgs = [
            SystemMessage(
                content=(
                    f"{SYSTEM_INSTRUCTIONS}\n\n"
                    f'{summary_context}\n'
                    f"<CONTEXTO_RECUPERADO>\n{context}\n</CONTEXTO_RECUPERADO>"
                )
            )
        ] + messages
        response = llm.invoke(prompt_msgs)
        return {"messages": [response]}

    def unknown_answer(state:State):
        
        if not state['has_context']:
            return{
                'messages':[AIMessage(content='Lo siento, no tengo esa información')],
                'valid_answer':False
            }

        # ✅ Correcto
        last_ai = next(
            (m for m in reversed(state['messages']) if isinstance(m, AIMessage)),
            None
        )
        flag_question = last_ai is not None and 'Lo siento, no tengo esa información' in last_ai.content
        return {'valid_answer': not flag_question}
        

    def should_summarize(state: State):
    
        if (
            len(state["messages"]) >= 10 
        and state.get('has_context',False)
        and state.get('valid_answer',False)
        ):
            return "summarize"
        return END


    def summarize_memory(state: State):

        messages = state["messages"]
        summary = state.get("summary", "")

        if len(messages) < 10:
            return {}

        messages_to_summarize=messages[:-4]
        prompt_msgs = [SystemMessage(content=(
        f"resumen actual: {summary} \n\n"
        f'Eres un sistema que hace resumen de una conversacion utilizando\n'
        f'el resumen dado anteriormente y las conversaciones dadas a continuacion'
        ))]+messages_to_summarize
        
        new_summary = llm.invoke(prompt_msgs)
        
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]

        return {
            "summary": new_summary.content,
            "messages": messages_to_remove
        }

    



    
    builder = StateGraph(State)
    builder.add_node("reformulate", reformulate_node)
    builder.add_node("context", context_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("summarize", summarize_memory)
    builder.add_node('no_answer',unknown_answer)

    builder.add_edge(START, "reformulate")
    builder.add_edge("reformulate", "context")

    builder.add_conditional_edges(
        source='context',
        path=should_continue,
        path_map={
            'chatbot':'chatbot',
            'no_answer':'no_answer'
        }
    )

    builder.add_edge("chatbot", 'no_answer')

    builder.add_conditional_edges(
        source='no_answer',
        path=should_summarize,
        path_map={
            'summarize':'summarize',
            END:END
        }
    )
    builder.add_edge("summarize", END)


    return builder.compile(checkpointer=get_redis_saver())



