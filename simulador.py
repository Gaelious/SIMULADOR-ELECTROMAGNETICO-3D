import numpy as np
import plotly.graph_objects as go
import streamlit as st # NUEVO MOTOR WEB

# ==========================================
# 0. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador EM", layout="wide")
st.title("⚡ Simulador de Campos Electromagnéticos 3D")

# Creamos un panel lateral para los controles
st.sidebar.header("Panel de Control")
# Un deslizador para mover la carga positiva en el eje X (de -2 a 2)
posicion_x_positiva = st.sidebar.slider("Posición Carga Positiva (X)", -2.0, 2.0, -1.0)
posicion_x_negativa = st.sidebar.slider("Posición Carga Negativa (X)", -2.0, 2.0, 1.0)

# ==========================================
# 1. EL MOTOR FÍSICO (Exactamente igual que antes)
# ==========================================
def calcular_campo_3d(q, pos_q, X, Y, Z):
    ke = 8.99e9 
    rx, ry, rz = X - pos_q[0], Y - pos_q[1], Z - pos_q[2]
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    r[r == 0] = 1e-9 
    return ke*q*rx/r**3, ke*q*ry/r**3, ke*q*rz/r**3

x, y, z = np.linspace(-3, 3, 20), np.linspace(-3, 3, 20), np.linspace(-3, 3, 20)
X, Y, Z = np.meshgrid(x, y, z)

# ATENCIÓN AQUÍ: Usamos las variables del slider en lugar de números fijos
E1x, E1y, E1z = calcular_campo_3d(1e-9, [posicion_x_positiva, 0, 0], X, Y, Z)
E2x, E2y, E2z = calcular_campo_3d(-1e-9, [posicion_x_negativa, 0, 0], X, Y, Z)

Ex, Ey, Ez = E1x + E2x, E1y + E2y, E1z + E2z

# ==========================================
# 2. EL MOTOR GRÁFICO (Casi igual que antes)
# ==========================================
E_magnitud = np.sqrt(Ex**2 + Ey**2 + Ez**2) + 1e-15
E_log = np.log10(E_magnitud + 1)

Ex_plot = (Ex / E_magnitud) * E_log
Ey_plot = (Ey / E_magnitud) * E_log
Ez_plot = (Ez / E_magnitud) * E_log

fig = go.Figure(data=go.Cone(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    u=Ex_plot.flatten(), v=Ey_plot.flatten(), w=Ez_plot.flatten(),
    colorscale='Plasma'
))

# Pintamos las cargas en su nueva posición dinámica
fig.add_trace(go.Scatter3d(
    x=[posicion_x_positiva, posicion_x_negativa], y=[0, 0], z=[0, 0],
    mode='markers+text', marker=dict(size=12, color=['red', 'blue']),
    text=['+q', '-q'], textposition="top center"
))

fig.update_layout(scene=dict(aspectmode="data"), height=700)

# ==========================================
# 3. MOSTRAR EN LA WEB (La magia de Streamlit)
# ==========================================
# En lugar de guardar un HTML, le decimos a Streamlit que pinte la figura
st.plotly_chart(fig, use_container_width=True)