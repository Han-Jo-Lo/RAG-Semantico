@echo off
title Asistente de Consulta RAG - Rappi
cls

echo =========================================
echo    Iniciando Asistente de Consulta RAG   
echo =========================================

:: 1. Iniciar el contenedor de Redis en Docker
echo [1/3] Iniciando memoria intermedia (Docker Redis)...
docker start redis-stack
if %errorlevel% neq 0 (
    echo [ERROR] Docker no esta corriendo o el contenedor 'redis-stack' no existe.
    echo Por favor, abre Docker Desktop e intenta de nuevo.
    pause
    exit
)

:: 2. Crear y/o activar el entorno virtual de Windows
echo [2/3] Verificando entorno de Python...
if not exist venv (
    echo Creando entorno virtual nuevo para Windows...
    python -m venv venv
    call venv\Scripts\activate
    echo Instalando librerias del proyecto (esto tomara un momento la primera vez)...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: 3. Obtener la IP local en Windows para la red
echo [3/3] Calculando IP de la red local...
for /f "tokens=2 del its=:" %%a in ('ipconfig ^| find "IPv4"') do (
    set IP_LOCAL=%%a
    goto :break
)
:break
:: Limpiar espacios en blanco de la IP
set IP_LOCAL=%IP_LOCAL: =%

echo =========================================
echo  ¡APLICACION LISTA PARA USAR!
echo =========================================
echo  - En esta computadora entra a: http://localhost:8501
echo  - Tus companeros en la red entran a: http://%IP_LOCAL%:8501
echo =========================================
echo Presiona Ctrl+C en esta ventana negra para apagar la aplicacion.
echo -----------------------------------------

:: 4. Lanzar Streamlit expuesto a la red local
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
pause