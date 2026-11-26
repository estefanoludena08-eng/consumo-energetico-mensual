# Predicción del consumo energético

Proyecto final del curso **Inteligencia Artificial Aplicada** (Universidad de Lima, 2026-2).
ODS 7: Energía asequible y no contaminante · ODS 12: Producción y consumo responsables.

Integrantes: Alexia Duffó, Angel Fajardo, Estefano Ludeña y Luis Fabián Rodriguez.

## Datos
*Appliances Energy Prediction*: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81-97. https://doi.org/10.1016/j.enbuild.2017.01.083

## Archivos
| Archivo | Descripción |
|---|---|
| `energydata_complete.csv` | Datos originales (19 735 registros, 29 variables) |
| `Modelo_ia_prediccion_consumo.ows` | Workflow de Orange 3.41 (preparación, exploración, 6 modelos y evaluación) |
| `app.py` | Prototipo web *WattCast* (Streamlit): estima el consumo mensual (kWh/mes) y su nivel |
| `requirements.txt` | Bibliotecas necesarias para el prototipo |
| `figuras/` | Figuras del informe y capturas de Orange |
| `LEEME.txt` | Instrucciones detalladas |

Modelo entrenado (Random Forest, ~53 MB): [PEGAR ENLACE DE GOOGLE DRIVE]. También se vuelve a generar al abrir el workflow en Orange.

## Cómo ejecutar el workflow en Orange
1. Descargar el repositorio completo (botón **Code → Download ZIP**) y descomprimirlo.
2. Abrir `Modelo_ia_prediccion_consumo.ows` en Orange 3.41 o superior.
3. Hacer doble clic en el widget **Datos (energydata_complete)** y presionar **Reload**.
4. Esperar 1 a 3 minutos. Los resultados aparecen en el widget **MAE, RMSE, R2**.

## Cómo ejecutar el prototipo
```
pip install -r requirements.txt
streamlit run app.py
```
Se abre en http://localhost:8501.

## Resultados (25 % de prueba, 4933 registros)
| Modelo | RMSE (Wh) | MAE (Wh) | R² |
|---|---|---|---|
| Random Forest | 68,44 | 32,78 | 0,518 |
| AdaBoost | 71,81 | 28,49 | 0,470 |
| SVM | 80,92 | 33,50 | 0,327 |
| Gradient Boosting | 82,18 | 44,47 | 0,305 |
| Red neuronal | 83,71 | 47,45 | 0,279 |
| Regresión lineal | 89,90 | 51,73 | 0,169 |

## Uso de inteligencia artificial
El código del prototipo se generó con apoyo de Claude (Anthropic), según lo declarado en el informe.
