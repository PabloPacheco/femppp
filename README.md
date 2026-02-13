# femppp

**femppp** es un código educativo de Elementos Finitos (FEM) desarrollado en Python para la enseñanza de la mecánica de sólidos en régimen de **esfuerzo plano**. El código está diseñado para ser ejecutado en **Jupyter Lab** y hace uso intensivo de `numpy` y `scipy.sparse` para un ensamblaje eficiente.

---

## Fundamentos Teóricos

### 1. Estado de Esfuerzo Plano

En problemas de esfuerzo plano se asume que no hay fuerzas en la dirección $z$ y que el espesor es pequeño, por lo que los esfuerzos fuera del plano son nulos ($\sigma_{zz} = \tau_{xz} = \tau_{yz} = 0$). El estado de esfuerzos se reduce a:

$$
\{\sigma\} = \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \tau_{xy} \end{Bmatrix}
$$

Las deformaciones asociadas son:

$$
\{\epsilon\} = \begin{Bmatrix} \epsilon_{xx} \\ \epsilon_{yy} \\ \gamma_{xy} \end{Bmatrix}
$$

con las relaciones cinemáticas lineales:

$$
\epsilon_{xx} = \frac{\partial u}{\partial x},\quad 
\epsilon_{yy} = \frac{\partial v}{\partial y},\quad 
\gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}
$$

### 2. Ley de Hooke para Esfuerzo Plano

La relación constitutiva $\{\sigma\} = [D]\{\epsilon\}$ utiliza la matriz de elasticidad:

$$
[D] = \frac{E}{1-\nu^2} \begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix}
$$

donde $E$ es el módulo de Young y $\nu$ el coeficiente de Poisson.

### 3. Principio de Mínima Energía Potencial

El método de elementos finitos implementado se basa en la formulación débil derivada del principio de mínima energía potencial total. Para un sólido elástico lineal, la energía potencial total $\Pi$ es:

$$
\Pi(\mathbf{U}) = \Lambda(\mathbf{U}) - W_{\text{ext}}(\mathbf{U})
$$

donde:
- $\Lambda$ es la energía de deformación interna,
- $W_{\text{ext}}$ es el trabajo realizado por las fuerzas externas,
- $\mathbf{U}$ es el vector de desplazamientos nodales (incógnitas).

La condición de equilibrio corresponde al punto estacionario de $\Pi$, lo que conduce al sistema de ecuaciones global $[K]\{U\} = \{F\}$.

---

## Elemento Triangular Lineal (CST)

El dominio se discretiza mediante triángulos de tres nodos. En cada elemento los desplazamientos se interpolan linealmente a partir de los valores nodales usando **funciones de forma** $S_i$, $S_j$, $S_k$:

$$
\begin{Bmatrix} u \\ v \end{Bmatrix} = 
\begin{bmatrix}
S_i & 0 & S_j & 0 & S_k & 0 \\
0 & S_i & 0 & S_j & 0 & S_k
\end{bmatrix}
\begin{Bmatrix} U_{ix} \\ U_{iy} \\ U_{jx} \\ U_{jy} \\ U_{kx} \\ U_{ky} \end{Bmatrix}
$$

Las funciones de forma en coordenadas cartesianas son:

$$
S_i = \frac{1}{2A}(\alpha_i + \beta_i x + \delta_i y), \quad
S_j = \frac{1}{2A}(\alpha_j + \beta_j x + \delta_j y), \quad
S_k = \frac{1}{2A}(\alpha_k + \beta_k x + \delta_k y)
$$

donde $A$ es el área del triángulo. Los coeficientes geométricos se calculan a partir de las coordenadas nodales $(x_i,y_i)$, $(x_j,y_j)$, $(x_k,y_k)$:

$$
\begin{aligned}
\alpha_i &= x_j y_k - x_k y_j, \quad &
\beta_i  &= y_j - y_k, \quad &
\delta_i &= x_k - x_j, \\
\alpha_j &= x_k y_i - x_i y_k, \quad &
\beta_j  &= y_k - y_i, \quad &
\delta_j &= x_i - x_k, \\
\alpha_k &= x_i y_j - x_j y_i, \quad &
\beta_k  &= y_i - y_j, \quad &
\delta_k &= x_j - x_i.
\end{aligned}
$$

Estas funciones satisfacen $S_i + S_j + S_k = 1$ en todo el elemento.

### 3.1 Matriz de Deformación $[B]$

Las deformaciones se obtienen derivando el campo de desplazamientos, resultando en una matriz constante para cada elemento:

$$
\{\epsilon\} = [B]\{U\}, \quad
[B] = \frac{1}{2A} \begin{bmatrix}
\beta_i & 0 & \beta_j & 0 & \beta_k & 0 \\
0 & \delta_i & 0 & \delta_j & 0 & \delta_k \\
\delta_i & \beta_i & \delta_j & \beta_j & \delta_k & \beta_k
\end{bmatrix}
$$

### 3.2 Matriz de Rigidez del Elemento

La energía de deformación se expresa como:

$$
\Lambda^{(e)} = \frac{1}{2} \{U\}^T \left( \int_V [B]^T [D] [B] \, dV \right) \{U\}
$$

Dado que $[B]$ y $[D]$ son constantes en el elemento, la integral de volumen se reduce a:

$$
[K]^{(e)} = [B]^T [D] [B] \, V, \qquad V = A \cdot t
$$

siendo $t$ el espesor del elemento.

---

## Vectores de Carga

### 4.1 Fuerzas de Cuerpo (Gravedad, etc.)

Para una fuerza por unidad de volumen $\{b\} = [b_x, b_y]^T$, la carga nodal equivalente se distribuye uniformemente entre los tres nodos:

$$
\{F\}^{(e)}_{\text{body}} = \frac{V}{3} \begin{Bmatrix} b_x \\ b_y \\ b_x \\ b_y \\ b_x \\ b_y \end{Bmatrix}
$$

### 4.2 Fuerzas Puntuales

Se aplican directamente en los grados de libertad correspondientes mediante el método `apply_point_loads`.

### 4.3 Fuerzas Distribuidas en Bordes

Para una carga superficial $\{t\} = [t_x, t_y]^T$ actuando sobre una arista del elemento (por ejemplo, la arista $ij$ de longitud $L_{ij}$), el trabajo externo conduce a:

$$
\{F\}^{(e)}_{\text{edge}} = \frac{t \, L_{ij}}{2} \begin{Bmatrix} t_x \\ t_y \\ t_x \\ t_y \\ 0 \\ 0 \end{Bmatrix}
$$

(análogamente para las otras aristas). Esta expresión corresponde a una distribución lineal de las fuerzas nodales equivalentes.

---

## Ensamblaje Global

El proceso de ensamblaje sigue los pasos clásicos:

1. Se calculan las matrices elementales $[K]^{(e)}$ y los vectores de carga.
2. Se determina la correspondencia entre grados de libertad locales y globales.
3. Se ensambla la matriz global $[K]$ en formato **sparse** (`lil_matrix` inicialmente, luego convertida a `csr_matrix` para eficiencia).
4. Se aplican condiciones de borde **Dirichlet** (desplazamientos prescritos) mediante eliminación o partición del sistema.
5. Se resuelve el sistema lineal $[K_{ff}]\{u_f\} = \{f_f\}$ usando un solver directo (`spsolve`) o iterativo (`cg`, `gmres`, `bicgstab`).

### 5.1 Ensamblaje Optimizado

Para mejorar el rendimiento, el ensamblaje de la matriz de rigidez se realiza de forma **vectorizada**: se recolectan todas las matrices $[B]$, $[D]$ y los vectores de DOFs en arrays de NumPy, y luego se construye una única matriz `coo_matrix` con todas las contribuciones.

---

## Ejemplos de Validación

El repositorio incluye ejemplos que reproducen benchmarks clásicos de la literatura:

- **Viga en voladizo con carga puntual en el extremo**: comparación con la solución analítica de Euler-Bernoulli.
- **Viga bi‑empotrada con carga de gravedad** (propuesta por el usuario).
- **Viga trapezoidal del benchmark NAFEMS** (caso de gravedad), validando los valores de esfuerzo cortante y normal en puntos específicos.

Estos ejemplos están documentados en notebooks de Jupyter y permiten visualizar la convergencia de la solución al refinar la malla.

---

## Estructura del Código

- **`FEMProblem`**: clase principal que gestiona el ensamblaje, las condiciones de borde y la solución.
- **`TriangularElement`**: clase que define el elemento CST (geometría, matriz $[B]$, matriz de rigidez).
- **`Tessellation`**: clase auxiliar para generar mallas estructuradas a partir de una nube de puntos (basada en `Delaunay`).
- **Módulos de utilidad**: funciones para aplicar cargas distribuidas, de cuerpo, etc.

---

## Referencias

1. Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2013). *The Finite Element Method: Its Basis and Fundamentals*. Butterworth-Heinemann.
2. Reddy, J. N. (2006). *An Introduction to the Finite Element Method*. McGraw-Hill.
3. NAFEMS (1987). *Linear Statics Benchmarks Vol. 1* (caso de viga trapezoidal).

