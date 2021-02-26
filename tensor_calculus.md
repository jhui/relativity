$$
{d^2 x^\mu \over dλ^2} + \Gamma^\mu_{\alpha\beta} {dx^\alpha \over dλ} {dx^\beta \over dλ} = 0$$

$$
\begin{align}
& \vec{a} = - ∇ 𝜙 \\
& F = \frac{GMm}{r^2}\vec{e_r},
\quad
g = \frac{F}{m} = \frac{GM}{r^2}\vec{e_r}\\
& g = - ∇ 𝜙\\
& 𝜙 = -\frac{GM}{r}  
\\
&\iint_S g \cdot \hat{n} dA = \iint_S - \left( \frac{GM}{r^2} \vec{e_r} \right) \cdot (\vec{e_r}) dA =  - \frac{GM}{r^2} \iint_S dA =  -\frac{GM}{r^2} 4 \pi r^2 = - 4 \pi  GM
\\
& \iiint_{V} (∇ \cdot g) dV = \iint_S g  \cdot  \hat{n} dA \\
 \iiint_{V} (∇ \cdot g) dV & = -4 \pi  GM = -4 \pi  G \iiint_V ρ dV = \iiint_V -4 \pi  G ρ dV \\
 ∇ \cdot g & = -4 \pi  G ρ \\
- ∇ \cdot ∇𝜙 & = -4 \pi  G ρ \\
∇^2 𝜙 & = 4 \pi  G ρ \\

& f \text{ is a scalar field,}\\
& v \text{ is a vector field: } \vec{v} =  v^x \vec{e_x} + v^y
\vec{e_y}
+ v^z \vec{e_z}\\
\end{align}
$$
$$
\begin{align}
∇_{\vec{u}} f = ∇ f \cdot \vec{u}= u^x \frac{\partial f}{\partial x} +
u^y \frac{\partial f}{\partial y} +
u^z \frac{\partial f}{\partial z} \\
∇ f = \frac{\partial f}{\partial x}\vec{e_x} +
\frac{\partial f}{\partial y}\vec{e_y} +
\frac{\partial f}{\partial z}\vec{e_z} \\
∇ \cdot \vec{v} = \frac{\partial v^x}{\partial x} +
\frac{\partial v^y}{\partial y} +
\frac{\partial v^z}{\partial z} \\
∇ \times \vec{v} = \frac{\partial f}{\partial x} +
\frac{\partial f}{\partial y} +
\frac{\partial f}{\partial z} \\
∇ \cdot \vec{v} < 0 \quad ∇ \cdot \vec{v} = 0 \quad ∇ \cdot \vec{v}> 0
\end{align}

$$

$$
\begin{align}

x & = r \cos θ, \quad y = r \sin θ \\
r & = \sqrt{x^2 + y^2}, \quad θ = \arctan \big(\frac{y}{x} \big) \\
\vec{ e_{r}} & =
\frac{\partial x}{\partial r} \vec{ e_{x}} + \frac{\partial x}{\partial r}  \vec{ e_{y}} \\
\vec{ e_{r}} & =
\cos θ \vec{ e_{x}} + \sin θ \vec{ e_{y}} \\
\vec{ e_{θ}} & =
\frac{\partial x}{\partial θ}  \vec{ e_{x}} + \frac{\partial y}{\partial θ} \vec{ e_{y}} \\
\vec{ e_{θ}} & =
-r \sin θ \vec{ e_{x}} + r \cos θ \vec{ e_{y}} \\
\end{align}

$$

$$
\frac{d \vec{R}}{d λ} =
\frac{\partial \vec{R}}{\partial c^i}
\frac{d c^i}{d λ}
= \frac{d c^i}{d λ} \vec{e_{c^i}} \quad \quad \text{since }
\frac{\partial \vec{R}}{\partial c^i} = \vec{e_{c^i}}
$$

$$
\begin{align}
\text{In general: } &\\
& ∇f = g^{ij} \frac{\partial f}{\partial c^i}\vec{e_{c^j}} \\
\text{If basis vectors are orthgonal, }\\
& ∇f = \frac{\partial f}{\partial c^i}\vec{e_{c^i}} \\
\text{For example, in Cartesian coordinates: } \\
& ∇f =
\begin{pmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y} \\
\frac{\partial f}{\partial z} \\
\end{pmatrix} \\
\end{align}
$$
