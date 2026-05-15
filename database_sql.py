import sqlite3
import pandas as pd
import io
from datetime import datetime

class SQLite_Manager:
    def __init__(self,base_datos_vectorial,sql_db_name):
        self.base_datos_vectorial=base_datos_vectorial
        self.sql_db_name=sql_db_name
    
    def registrar_pregunta(self,pregunta, usuario):
        """Guarda la pregunta en SQLite."""
        with sqlite3.connect(self.sql_db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preguntas_fallidas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT,
                    usuario TEXT,
                    base_datos_vectorial TEXT,
                    pregunta TEXT
                )
            ''')
            # Corregido: Usamos el nombre exacto de la columna 'base_datos_vectorial'
            cursor.execute(
                "INSERT INTO preguntas_fallidas (fecha, usuario, base_datos_vectorial, pregunta) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), usuario,self.base_datos_vectorial, pregunta)
            )
            conn.commit()

    def preparar_excel_descarga(self):
        """
        Lee de SQLite y genera los bytes para el botón de Streamlit.
        """
        try:
            with sqlite3.connect(self.sql_db_name) as conn:
                # Filtramos usando el nombre correcto de la columna
                query = "SELECT * FROM preguntas_fallidas WHERE base_datos_vectorial = ?"
                df = pd.read_sql_query(query, conn, params=(self.base_datos_vectorial,))
                
                if df.empty:
                    return None # Si no hay datos, devolvemos None para no mostrar el botón

                # Creamos el archivo en memoria (BytesIO)
                output = io.BytesIO()
                # Usamos un context manager para asegurar que el Excel se cierre y escriba bien
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Gaps de Información')
                
                return output.getvalue() # Esto devuelve los bytes que Streamlit necesita
                
        except Exception as e:
            print(f"Error al preparar Excel: {e}")
            return None