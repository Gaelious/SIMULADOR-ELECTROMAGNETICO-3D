import numpy as np
import pyvista as pv

# ==========================================
# 1. CREAR EL ESPACIO DE TRABAJO (Malla 3D)
# ==========================================
# Creamos 108,000 puntos de cálculo (Tu GPU lo hará sin sudar)
x = np.linspace(-3, 3, 60)
y = np.linspace(-3, 3, 60)
z = np.linspace(-1.5, 1.5, 30)
grid = pv.RectilinearGrid(x, y, z)

# Extraemos las coordenadas matemáticas de cada punto
pts = grid.points
X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

# ==========================================
# 2. DEFINIR EL NÚCLEO EN "C" (Geometría)
# ==========================================
R = np.sqrt(X**2 + Y**2)
R_centro = 2.0 # El centro de la pista magnética

# Lógica Booleana: ¿Qué puntos forman el núcleo?
# 1. Es un anillo con radio entre 1.5 y 2.5, y altura Z entre -0.5 y 0.5
en_anillo = (R > 1.5) & (R < 2.5) & (np.abs(Z) < 0.5)

# 2. El entrehierro (Air Gap) está en la parte derecha (X>0) y es un corte estrecho (Y entre -0.2 y 0.2)
en_gap = (X > 0) & (np.abs(Y) < 0.2)

# 3. El núcleo de ferrita es el anillo MENOS la zona del entrehierro
es_nucleo = en_anillo & ~en_gap

# Guardamos esta información en la malla y extraemos el modelo 3D del núcleo
grid["material"] = es_nucleo.astype(float)
nucleo_mesh = grid.threshold(0.5, scalars="material")

# ==========================================
# 3. FÍSICA: CAMPO MAGNÉTICO Y DISPERSIÓN (Fringing)
# ==========================================
R_safe = np.where(R == 0, 1e-9, R)

# A. Campo Base: El flujo da vueltas en círculo (Vector tangente)
Bx = -Y / R_safe
By = X / R_safe
Bz = np.zeros_like(Z)

# B. Modelo matemático del Fringing Effect
# Usamos una campana de Gauss centrada en el entrehierro para saber cuándo empujar el campo
intensidad_dispersion = np.exp(-(Y**2) / 0.05) * (X > 0)

# Cuando nos acercamos al aire, los vectores sufren una expansión (divergencia)
# Empuje radial (se salen por los lados del núcleo)
Bx += intensidad_dispersion * (R - R_centro) * (X / R_safe) * 1.5
# Empuje axial (se salen por arriba y por abajo del núcleo)
Bz += intensidad_dispersion * Z * 1.5

# Guardamos el campo vectorial total en nuestro universo PyVista
grid["B_field"] = np.column_stack((Bx, By, Bz))

# ==========================================
# 4. GENERAR LOS TUBOS DE FLUJO (Streamlines)
# ==========================================
# Plantamos una "rejilla de semillas" cuadrada en la parte izquierda del núcleo.
# Desde aquí nacerán las líneas de flujo y viajarán por la física que hemos programado.
semillas = pv.Plane(
    center=(-2, 0, 0), direction=(0, 1, 0), 
    i_size=0.8, j_size=0.8, 
    i_resolution=7, j_resolution=7
)

# El motor C++ calcula el recorrido exacto de los campos
flujo = grid.streamlines_from_source(
    semillas,
    vectors="B_field",
    integration_direction="both", # Que fluya hacia adelante y hacia atrás
    max_steps=2000
)

# ==========================================
# 5. RENDERIZADO PROFESIONAL
# ==========================================
plotter = pv.Plotter()
plotter.background_color = "black"

# 1. Dibujamos los tubos de flujo electromagnético con textura térmica
plotter.add_mesh(flujo.tube(radius=0.015), cmap="plasma", show_scalar_bar=False)

# 2. Dibujamos el núcleo metálico semitransparente para ver el interior
plotter.add_mesh(nucleo_mesh, color="silver", opacity=0.25)

# Añadimos bordes y títulos
plotter.add_bounding_box(color="gray")
plotter.add_text("Efecto de Dispersión en Entrehierro (Fringing Effect)", font_size=12, color="white")

# ¡A renderizar en la GPU!
plotter.show()