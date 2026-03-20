import gradio as gr
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load('modelo_autos_RF.joblib')

# Función de predicción
def predecir_precio(mileage, engine, power, year, fuel, transmission):
    if fuel == "Eléctrico":
        transmission = "Automática"

    Trans_Automatic = 1 if transmission == "Automática" else 0
    Trans_Manual = 1 if transmission == "Manual" else 0
    Fuel_Diesel = 1 if fuel == "Diésel" else 0
    Fuel_Electric = 1 if fuel == "Eléctrico" else 0
    Fuel_Petrol = 1 if fuel == "Gasolina" else 0

    entrada = np.array([[mileage, engine, power, Trans_Automatic, Trans_Manual, 
                         year, Fuel_Diesel, Fuel_Electric, Fuel_Petrol]])
    
    precio_predicho = modelo.predict(entrada)[0]
    return f"💰 ₹{precio_predicho:,.2f}"

# Estilos CSS personalizados para escala de grises y selección activa
custom_css = """
body {background-color: white; color: black; font-family: 'Arial', sans-serif;}
.gradio-container {background-color: white !important;}
input, select, textarea {background-color: #F0F0F0 !important; color: black !important; border-radius: 5px; padding: 10px;}
button {background-color: #333333 !important; color: white !important; font-size: 16px; font-weight: bold; border-radius: 5px; padding: 10px 15px;}
button:hover {background-color: #666666 !important;}
.gradio-slider input[type="range"] {background-color: #F0F0F0 !important;}
.gradio-slider input[type="range"]::-webkit-slider-runnable-track {
    background: #D3D3D3 !important;
    height: 8px;
    border-radius: 5px;
}
.gradio-slider input[type="range"]::-webkit-slider-thumb {
    background: #333333 !important;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    border: none;
}

/* Cambiar el color de fondo de la selección */
.gradio-radio input[type="radio"]:checked + label {
    background-color: #333333 !important;
    color: white !important;
    border-radius: 5px;
    padding: 10px;
}

.gradio-radio input[type="radio"]:not(:checked) + label {
    background-color: #F0F0F0 !important;
    color: black !important;
    border-radius: 5px;
    padding: 10px;
}
"""

# Interfaz con diseño en escala de grises
with gr.Blocks(css=custom_css) as interfaz:
    gr.Markdown("# 🚗 Predicción del Precio de Autos Usados")
    gr.Markdown("### Ingresa las características del auto para estimar su precio.")

    with gr.Row():
        mileage = gr.Number(label="Kilometraje (km/l)", value=15, precision=0)  # Solo enteros
        engine = gr.Number(label="Tamaño del motor (cc)", value=1500, precision=0)  # Solo enteros
    
    with gr.Row():
        power = gr.Number(label="Potencia (bhp)", value=100, precision=0)  # Solo enteros
        year = gr.Slider(1990, 2024, value=2015, label="Año del Auto", step=1)

    fuel = gr.Radio(["Diésel", "Eléctrico", "Gasolina"], label="Combustible", value="Gasolina")
    transmission = gr.Radio(["Automática", "Manual"], label="Transmisión", value="Automática")

    # Desactivar opción "Manual" si el usuario selecciona "Eléctrico"
    def actualizar_transmision(fuel):
        return "Automática" if fuel == "Eléctrico" else gr.update()

    fuel.change(fn=actualizar_transmision, inputs=[fuel], outputs=[transmission])

    boton = gr.Button("Predecir Precio 💰")
    salida = gr.Textbox(label="Precio Estimado", interactive=False)

    boton.click(predecir_precio, inputs=[mileage, engine, power, year, fuel, transmission], outputs=salida)

# Lanzar aplicación
interfaz.launch()