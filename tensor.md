$$
\begin{align}
∆τ = \sqrt{\text{ten years}} \quad \text{from } (0, 0, 0, 0) \text{ to }(10, 0, 0, 0)\\

\quad \text{from } (0, 0, 0, 0) \text{ to }(10, 0, 0, 0)\\

∆s^2 & = -(∆ct)^2 + ∆x^2 + ∆y^2 + ∆z^2 \\
∆τ^2 & = (∆ct)^2 - ∆x^2 - ∆y^2 - ∆z^2 \\
∆s^2 & = -(∆ct)^2 + ∆x^2 \\
&= -γ^2\left(\left( ∆ct'- \frac{v∆x'}{c} \right)^2 +  \left( ∆x' - v∆t' \right)^2 \right) \text{(Apply Lorentz transformation from t to t')}\\
&= -γ^2 \left(∆ct' - \frac{v∆x'}{c} \right)^2
 +  γ^2 \left( ∆x' - v∆t' \right)^2\\
&= -γ^2\left( (∆ct')^2  - 2v∆x'∆t' +\frac{(v∆x')^2}{c^2} - ∆x'^2 + 2v∆x'∆t' - (v∆t')^2\right) \\
& =  -\frac{c^2}{c^2 - v^2}\left( c^2(∆t')^2  - v^2(∆t')^2 +\frac{v^2(∆x')^2}{c^2} - ∆x'^2  \right) \\
& = -(∆ct')^2 -\frac{ (v^2 - c^2)∆x'^2}{c^2 - v^2}  \\
& = -(∆ct')^2 + ∆x'^2 \\
& = ∆s'^2

\\
&\text{In rest frame: } S, \\
&\text{Head of the train: } (0, l) \\
&\text{Tail of the train: } (0, 0) \\

\end{align}

$$

$$
\begin{align}
& \text{For the } ct' \text{ axis, } x'=0 \text{ and } x = vt, \\
& \text{substitute them into }  x' = Ct + Dx,\\
& \text{we get: } C = -Dv \text{ , and }\\
& x' = Ct + Dx = -Dvt + Dx \\
& x' = D(x -vt) \\
& \text{Given } t' = A t + B x \text{ and } x = 0, \text{ then } t' = At,  \\
& \text{Given } x' = D(x -vt), x =0  \text{ and }  x' = -vt' , \text{ then } -vt' = -Dvt t' = Dt  \\
& \text{Let } F = \frac{B}{A} \\
& t' = A t + B x = A(t + \frac{B}{A}x) = A (t+ Fx)\\
& \text{Lets change A to γ: } A = γ \\
& t' = At + Bx = γ(t + Fx) \\
& x' = D(x -vt) = A(x -vt) = γ( -vt + x) \\
& \begin{pmatrix}
t' \\
x' \\
\end{pmatrix} = γ  
\begin{pmatrix}
1 & F \\
-v & 1 \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix} \\

& \begin{pmatrix}
t' \\
x' \\
\end{pmatrix} = γ_{v_1}  
\begin{pmatrix}
1 & F_{v_1} \\
-v_1 & 1 \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix}\\

&\begin{pmatrix}
t'' \\
x'' \\
\end{pmatrix} = γ_{v_2}γ_{v_1}  
\begin{pmatrix}
1 & F_{v_2} \\
-v_2 & 1 \\
\end{pmatrix}
\begin{pmatrix}
1 & F_{v_1} \\
-v_1 & 1 \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix} \\

&\begin{pmatrix}
t'' \\
x'' \\
\end{pmatrix} =
γ_{v_2}γ_{v_1}  
\begin{pmatrix}
1 - v_1 F_{v_2}  & F_{v_1} + F_{v_2}\\
-v_1 -v_2 & 1 - v_2 F_{v_1}  \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix}\\
& 1 - v_1 F_{v_2} = 1 - v_2 F_{v_1} \\
& \frac{v_1}{F_{v_1}}= \frac{v_2}{F_{v_2}} \\
& \frac{v_1}{F_{v_1}}= \frac{v_2}{F_{v_2}} = a\\
& F_v = \frac{v}{a} \\
& \begin{pmatrix}
t' \\
x' \\
\end{pmatrix} = γ  
\begin{pmatrix}
1 & \frac{v}{a} \\
-v & 1 \\
\end{pmatrix} \begin{pmatrix}
t \\
x \\
\end{pmatrix}
\\

& F_v = \frac{v}{a} \\
&\begin{pmatrix}
t'' \\
x'' \\
\end{pmatrix} =
γ_{v_2}γ_{v_1}  
\begin{pmatrix}
1 - v_1 F_{v_2}  & F_{v_1} + F_{v_2}\\
-v_1 -v_2 & 1 - v_2 F_{v_1}  \\
\end{pmatrix} \\
&\begin{pmatrix}
t'' \\
x'' \\
\end{pmatrix} =
γ_{v_2}γ_{v_1}  
\begin{pmatrix}
1 - \frac{v_1 v_2}{a} & \frac{v_1}{a} + \frac{v_2}{a}\\
-v_1 -v_2 & 1 - \frac{v_1 v_2}{a}  \\
\end{pmatrix} \\

&\text{Set } v_1 = - v_2 \\
&\begin{pmatrix}
t'' \\
x'' \\
\end{pmatrix} =
γ_{-v_1}γ_{v_1}  
\begin{pmatrix}
1 - \frac{(v_1)^2}{a} & 0\\
0 & 1 - \frac{(v_1)^2}{a}  \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix}\\
&\begin{pmatrix}
t \\
x \\
\end{pmatrix} =
γ_{-v}γ_{v}  
\begin{pmatrix}
1 + \frac{v^2}{a} & 0\\
0 & 1 + \frac{v^2}{a}  \\
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix}\\
& 1 = γ_{-v_1}γ_{v_1}  \left(1 + \frac{v^2}{a} \right)
\\
& γ_{-v_1} = γ_{v_1} \\
& 1 = (γ_v)^2  \left(1 + \frac{v^2}{a} \right) \\
& γ = \frac{1}{\sqrt{\left(1 + \frac{v^2}{a} \right)}} \\
& t' = γ (\frac{vx}{a} + t)  \\
& x' = γ (x - vt) \\
& γ = \frac{1}{\sqrt{\left(1 + \frac{v^2}{a} \right)}} \\
& a = \frac{v}{F_v} \\
& a = - c^2 \\

&\begin{pmatrix}
t' \\
x' \\
\end{pmatrix} = γ  
\begin{pmatrix}
1 & -\frac{v}{c^2} \\
-v & 1 \\
\end{pmatrix} \begin{pmatrix}
t \\
x \\
\end{pmatrix} \quad \text{ where }
γ = \frac{1}{\sqrt{\left(1 - \frac{v^2}{c^2} \right)}} \\
& t' = γ\left( t - \frac{vx}{c^2} \right) \\
& x' = γ\left( x-vt \right) \\
& \text{if } x = ct\\
& t' = γ\left( t - \frac{vct}{c^2} \right) = γ\left( t - \frac{vt}{c} \right) \\
& ct' = γ\left( ct - vt \right) = γ\left( x - vt \right) = x'\\
\\

& L = γ L_0 \\

\end{align}
$$

$$
x' = Ct + Dx \\
\text{we get: }
\begin{align}
t' & = γ(t - \frac{vx}{c^2}) \\
x' & = γ(x - vt) \\
\begin{pmatrix}
t' \\
x' \\
\end{pmatrix} & = γ  
\begin{pmatrix}
1 & - \frac{v}{c^2} \\
-v & 1 \\
\end{pmatrix}

\begin{pmatrix}
t \\
x \\
\end{pmatrix}
\\
\end{align}
$$

$$
c = \frac{distance}{time} \\
\begin{pmatrix}
t' \\
x' \\
\end{pmatrix} =  
\begin{pmatrix}
A & B \\
C & D \
\end{pmatrix}
\begin{pmatrix}
t \\
x \\
\end{pmatrix}
\\
t' = A t + B x \\
x' = C t + D x \\

\sqrt{ \left(v t \right)^2 + L^2}\\
\\
\begin{align}
\sqrt{1+v^2} \\
time & = distance/speed \\
t' & = \frac{L}{c'} \\
t & = \frac{\sqrt{ ( v t )^2 + L^2}}{c} \\
 L^2 & = (ct)^2 - (vt)^2  \\
c & = c' = 1 \quad\quad  (\text{not } c = c' + v )\\
\frac{L}{ t'} &= \frac{\sqrt{ ( v t )^2 + L^2}}{t} \\
\frac{1}{ t'} &= \frac{\sqrt{ \frac{( v t )^2}{L^2} + 1}}{t} \\
\frac{t}{ t'} &= \sqrt{ \frac{( v t )^2}{t^2(1-v^2)} + 1} =  \sqrt{ \frac{( v )^2}{(1-v^2)} + 1} = \sqrt{ \frac{1}{(1-v^2)}}
 \\
t & = \sqrt{ \frac{1}{(1-v^2)}} t' = γ t' \quad \quad \text{where } γ = \sqrt{ \frac{1}{(1-v^2)}}\\
 \\
x' &= x - vt \\
y' &= y \\
z' &= z \\
t' &= t \\
\frac{\partial S(p)}{\partial {a^m}}
\quad \quad \quad \quad

F & = ma \\
\frac{d x'}{d t} & = \frac{d x}{d t} -V \\
\frac{d^2 x'}{d t^2} & = \frac{d^2 x}{d t^2} \\
F & = ma  = m \frac{d^2 x'}{d t^2} = m \frac{d^2 x}{d t^2} \\

\frac{\partial S(p)}{\partial {b^m}}
=
\frac{\partial S(p)}{\partial a^n}
\frac{\partial a^n}{\partial b^m}
\quad \quad \quad \quad
v'_m = \frac{\partial a^n}{\partial b^m} v_n \\
\text{Gradient} \\
\text{Apply chain rule} \\
\text{Covariant components}  \\


\\
v^m & = [v^1, v^2, v^3,  \cdots, v^N] \\
v'^k & = \sum_m \frac{\partial b^k}{\partial a^m} v^m \\
v'^k & =  \frac{\partial b^k}{\partial a^m} v^m \\
v'_m & =  \frac{\partial a^k}{\partial b^m} v_k \\
\text{Contravariat vector's components} \\
\text{Conavariat vector's components} \\
v'^k & =  \frac{\partial b^k}{\partial a^k} v^k \\
16 a^i & = b^i\\
\frac{\partial b^i}{\partial a^i} & = 16
\\
v'^i & =  \frac{\partial b^i}{\partial a^i} v^i  = 16 v^i
\end{align}
$$

$$
g_{ij}v^k \delta^i_k \\
g_{ij}v^i  \\
\frac{\partial \vec{R}}{\partial λ} =
\begin{pmatrix}
0 \\
1 \\
\end{pmatrix}_{\frac{\partial \vec{R}}{\partial r}, \frac{\partial \vec{R}}{\partial θ}}
=
\begin{pmatrix}
- \sin  λ\\
\cos λ \\
\end{pmatrix}
_{\frac{\partial \vec{R}}{\partial x}, \frac{\partial \vec{R}}{\partial y}}
$$

$$
\begin{align}
\frac{d \vec{R}}{d λ} &=
\frac{d a^i}{d λ}

 \frac{\partial \vec{R}}{\partial a^i}  
\\
 &=
 \frac{d a^i}{d λ} \vec{e_{a^i}}
\end{align}
\\
\frac{\partial p^j}{\partial λ} =
\frac{\partial p^j}{\partial c_i} \frac{\partial c^i}{\partial λ}

=B^j_i \frac{\partial c^i}{\partial λ} \\
\vec{u} \\
\vec{v} = v^i \vec{e_i} \\
df = \frac{\partial f}{\partial x} dx +
 \frac{\partial f}{\partial y} dy  \\
∇_{\vec{v}}f =\frac{\partial f}{\partial \vec{v}} = \vec{v} \cdot ∇f
\\ \quad f \\
\text {In Cartesian Coordinates, }\\
\frac{d T}{d λ} = \frac{\partial T}{\partial q_i}  \frac{d q_i}{d λ} \\
\frac{d T}{d t} = \frac{\partial T}{\partial x}  \frac{d x}{d t} +
\frac{\partial T}{\partial y}  \frac{d y}{d t} \\

i ∈ \{1, 2, 3\} \quad  j ∈ \{1, 2, 3\} \\
v^i = u^i  \quad or \quad
w_j = x_j \\
v^1 = u^1 \\
v^2 = u^2 \\
v^3 = v^3 \\
w_1 = x_1 \\
w_2 = x_2 \\
w_3 = x_3 \\

F =
\begin{pmatrix}
F^1_1 & F^1_2 \\
F^2_1 & F^2_2 \\
\end{pmatrix} \\
B =
\begin{pmatrix}
B^1_1 & B^1_2 \\
B^2_1 & B^2_2 \\
\end{pmatrix}
$$

$$
\begin{align}
f(x, y) &= 2x^2 + xy - 1 \\
(x, y) & = (t, t^2 -1) \\
\frac{d f}{d t} & = \frac{\partial f}{\partial x} \frac{d x}{d t}
+
\frac{\partial f}{\partial y} \frac{d y}{d t}
\\
 \frac{\partial f}{\partial x } & = 4x + y, \quad \frac{d x}{d t} = 1\\
   \frac{\partial f}{\partial y } &= x,
\quad \frac{d y}{d t} = 2t
    \\

\end{align}
$$

$$
V = V^m \vec{e_m} \\
V_n = V \cdot \vec{e_n} \\
\|v\|^2 = v_i v^i \\
\|v\|^2 = (v^1 \vec{e_1} + v^2 \vec{e_2}) (v^1 e_1 + v^2 e_2) \\
∇T(p) =
\begin{bmatrix}
\frac{\partial T(p)}{\partial x}  \\
\frac{\partial T(p)}{\partial y}  \\
\frac{\partial T(p)}{\partial z}  \\
\end{bmatrix}
$$

$$
\begin{align}
\|\vec{v}\|^2 & = (v^1 \vec{e_1} + v^2 \vec{e_2}) \cdot (v^1 \vec{e_1} + v^2 \vec{e_2}) \\
& = (v^1)^2 \vec{e_1} \cdot \vec{e_1} + 2 v^1v^2 \vec{e_1} \cdot \vec{e_2} +
(v^2)^2 \vec{e_2} \cdot \vec{e_2} \\
\end{align}
$$

$$
W = g_{μν} F^μ x^ν\\
dY^m = \frac{\partial Y^m}{\partial X^p} dX^P \\
dY^1 = \frac{\partial Y^1}{\partial X^1} dX^1 \\

\frac{d \arctan(u)}{du} = \frac{1}{1+u^2} \\
$$
$$
\frac{\partial} {\partial x^i}  = \vec{e_{x^i}} \\
L =
$$
$$
\sum_{β=1}^n \sum_{γ=1}^n A^α_{β}B^β_{γ}C^γ_{δ} → A^α_{β}B^β_{γ}C^γ_{δ} = D^γ_{δ}
$$
$$
\frac{\partial \vec{R}}{\partial a^j} =
\frac{\partial b^i}{\partial a^j}
\frac{\partial \vec{R}}{\partial b^i} \\
\frac{\partial \vec{R}}{\partial b^i} =
\frac{\partial a^j}{\partial b^i}
\frac{\partial \vec{R}}{\partial a^j} \\
$$

$$
\frac{\partial a^i}{\partial λ} =
\frac{\partial a^i}{\partial b^j}
\frac{\partial b^j}{\partial λ} \\
\frac{\partial b^j}{\partial λ} =
\frac{\partial b^j}{\partial a^i}
\frac{\partial a^i}{\partial λ} \\
F^i_j = \frac{\partial a^i}{\partial b^j}\\
B^j_i = \frac{\partial b^j}{\partial a^i}\\
$$

$$
\begin{align}
B^i_j &= \frac{\partial p^i}{\partial c^j}\\
 B &=
 \begin{pmatrix}
  \frac{\partial p^1}{\partial c^1} & \frac{\partial p^1}{\partial c^2}\\  
  \frac{\partial p^2}{\partial c^1} & \frac{\partial p^2}{\partial c^2}\\  
 \end{pmatrix} = J^{-1} \\
 &=
 \begin{pmatrix}
 \frac{x}{r} & \frac{y}{r} \\
 -\frac{y}{r^2} & \frac{x}{r^2} \\
 \end{pmatrix} \\

\frac{\partial p^1}{\partial c^1} &=
\frac{\partial r}{\partial x} = \frac{\partial}{\partial x}
 \sqrt{x^2+y^2}
 = \frac{x}{\sqrt{x^2+y^2}}
 = \frac{x}{r}
\\
\frac{\partial p^2}{\partial c^1} &=
\frac{\partial θ}{\partial x} = \frac{\partial}{\partial x}
 \arctan \left(\frac{y}{x}\right)
 = \frac{-y}{x^2 (1 + \frac{y^2}{x^2})}
 = -\frac{y}{r^2}
\\
B &=
\begin{pmatrix}
\frac{x}{r} & \frac{y}{r} \\
-\frac{y}{r^2} & \frac{x}{r^2} \\
\end{pmatrix} =
\begin{pmatrix}
\cos θ & \sin θ\\
-\frac{sin θ}{r} & \frac{\cos θ}{r} \\
\end{pmatrix} =
\begin{pmatrix}
\cos λ & \sin λ\\
-sin λ & \cos λ \\
\end{pmatrix}
\\


\end{align}

$$

$$
\frac{\partial \vec{R}}{\partial c^j} =
\frac{\partial p^i}{\partial c^j}
\frac{\partial \vec{R}}{\partial p^i} \\
\frac{\partial \vec{R}}{\partial p^i} =
\frac{\partial c^j}{\partial p^i}
\frac{\partial \vec{R}}{\partial c^j} \\
$$


$$
\frac{\partial c^i}{\partial λ} =
\frac{\partial c^i}{\partial p^j}
\frac{\partial p^j}{\partial λ} \\
\frac{\partial p^j}{\partial λ} =
\frac{\partial p^j}{\partial c^i}
\frac{\partial c^i}{\partial λ} \\
\frac{\partial p^j}{\partial λ} =
\frac{\partial p^j}{\partial c^i}
\frac{\partial c^i}{\partial λ} =
B \frac{\partial c^i}{\partial λ}
\\
$$

$$
\frac{\partial p}{\partial λ} =
\begin{pmatrix}
\cos λ & \sin λ \\
-\sin λ & \cos λ\\
\end{pmatrix}
\begin{pmatrix}
- \sin λ \\
 \cos λ\\
\end{pmatrix} =
\begin{pmatrix}
- \cos λ \sin λ + \cos λ \sin λ\\
  \sin^2 λ + \cos^2 λ \\
\end{pmatrix} =
\begin{pmatrix}  
0 \\
1 \\
\end{pmatrix} \\
\frac{\partial \vec{R}}{\partial λ} =
\begin{pmatrix}  
0 \\
1 \\
\end{pmatrix}_{\vec{e_r}, \vec{e_θ}}\\

$$

$$
\frac{\partial p^1}{\partial c^1} =
\frac{\partial r}{\partial x} =
\frac{x}{\sqrt{x^2 + y^2}} = \frac{x}{r} \\
$$

$$
(c^1, c^2) = (x, y) \\
(p^1, p^2) = (r, θ) \\
r = \sqrt{x^2 + y^2} \\
θ = \arctan \left( \frac{y}{x} \right) \\
\vec{e_x}, \vec{e_y} \\
\vec{e_r}, \vec{e_θ} \\
$$


$$
\begin{align}
R(λ) & = (x(λ), y(λ))
= (\cos λ, \sin λ) \\
& = (r(λ), θ(λ)) = (1, λ) \\
\end{align}
$$
$$
\begin{align}
R(λ) & = (\cos λ, \sin λ)_{e_x, e_y} \\
& = (1, λ)_{e_r, e_θ} \\
\end{align}
$$

$$
\begin{align}
R(λ) & = (\cos λ, \sin λ)_{e_x, e_y} \\
\frac{\partial R(λ)}{\partial λ} & =
\begin{pmatrix}
-\sin λ \\
 \cos λ \\
\end{pmatrix}
\\
& = (1, λ)_{e_r, e_θ} \\
\end{align}
$$

$$
\begin{align}
& ∂_α : \text{ partial derivative w.r.t } α
& c: \text{ the speed of light} \\
& ρ: \text{ electric charge density} \\
& j: \text{ electric current density} \\
& μ_0: \text{ permeability of free space} \\
& ϕ:  \text{ electric potential (a scalar potential)} \\
& A: \text{ magnetic vector potential (a vector potential) } \\
\text{four-potential} \\
\text{four-current}  \\
& \text{even: even number of two-element swaps to change } \\
& i, j, k, l \text{ to } (1, 2, 3, 4)
\end{align}\\
\begin{pmatrix}
0.4 & 0.5 \\
\end{pmatrix}  \\

0.4 x_1 + 0.5 x_2 ∈ V^* \quad \text{ Dual space} \\

0.6 x_1 + 0.2 x_2 ∈ V^* \\
\frac{\partial \vec{R}}{\partial x} = \lim_{Δ x \rightarrow 0} \frac{\vec{R}(x+ Δ x, y) - \vec{R}(x, y) }{Δ x} = \vec{e_x} \\
 \frac{\partial \vec{R}}{\partial x^i}  = \vec{e_{x^i}} \\
\text{cartesian: }\frac{\partial \vec{R}}{\partial x}  = \vec{e_{x}} , \frac{\partial \vec{R}}{\partial y}  = \vec{e_{y}} \\
\text{ polar: } \frac{\partial \vec{R}}{\partial r}  = \vec{e_{r}}
, \frac{\partial \vec{R}}{\partial θ}  = \vec{e_{θ}} \\
x = r\cos θ, \quad y = r\sin θ\\
\vec{ e_{r}} =
\cos θ \vec{ e_{x}} + \sin θ \vec{ e_{y}} \\
\vec{ e_{θ}} =
-r \sin θ \vec{ e_{x}} + r \cos θ \vec{ e_{y}} \\

\frac{\partial}{\partial x}  = \vec{e_{x}} \\

 F =
\begin{pmatrix}
\cos θ  & -r \sin θ \\
\sin θ & r \cos θ\\
\end{pmatrix} =
\begin{pmatrix}
x / \sqrt{x^2 + y^2}  & -y \\
y / \sqrt{x^2 + y^2} & x\\
\end{pmatrix} \\
B =
\begin{pmatrix}
x/r & y/r \\
-y / r^2 & x / r^2\\
\end{pmatrix} =
\begin{pmatrix}
x / \sqrt{x^2 + y^2}  & -y \\
y / \sqrt{x^2 + y^2} & x\\
\end{pmatrix} \\

F =
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial θ} \\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial θ} \\
\end{pmatrix}

\\

FB =
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial θ} \\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial θ} \\
\end{pmatrix}
\begin{pmatrix}
\frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} \\
\frac{\partial θ}{\partial x} & \frac{\partial θ}{\partial y} \\
\end{pmatrix} = T
\\
\text {For } a =  (x, y) , \quad b =(r, θ)\\
T^i_k =
 \frac{\partial a^i}{\partial b^j}  \frac{\partial b^j}{\partial a^k}
 =  \frac{d a^i}{d a^j}  =\delta^i_k
\\
R \\
\lambda \\
\frac{d \vec{R}}{d λ}
=
\frac{\partial \vec{R}}{\partial b^i}
\frac{d b^i}{d λ}
=
\frac{d b^i}{d λ}
\frac{\partial \vec{R}}{\partial b^i}
= \frac{d b^i}{d λ} \vec{e_{b_i}}
\\
\frac{d \vec{R}}{d λ}
=
\frac{\partial \vec{R}}{\partial a^i}
\frac{d a^i}{d λ}
=
\frac{d a^i}{d λ}
\left(
  \frac{\partial \vec{R}}{\partial b^i}
  \frac{\partial b^i}{\partial a^i}
\right)
=
\left(

\frac{d a^i}{d λ}
\frac{\partial b^i}{\partial a^i}
\right)
\frac{\partial \vec{R}}{\partial b^i}
=
\left(

\frac{d a^i}{d λ}
\frac{\partial b^i}{\partial a^i}
\right)
 \vec{e_{b_i}}
 \\
 \frac{d b^i}{d λ} =
 \frac{\partial b^i}{\partial a^i}
 \frac{d a^i}{d λ}
\\
r = \sqrt{x^2 + y^2} \\
θ = arctan \left( \frac{y}{x} \right)
\\
\begin{pmatrix}
0 \\ 1 \\
\end{pmatrix} \\
\frac{d R}{d \lambda } \\
R(x, y) =
\begin{pmatrix}
\cos \lambda &  \sin \lambda \\
\end{pmatrix} \\
\frac{d R}{d λ}
=
\frac{\partial R}{\partial a^i}
\frac{d a^i}{d λ}
=  \frac{d x}{d λ}
\frac{\partial R}{\partial x}
+  \frac{d y}{d λ}
\frac{\partial R}{\partial y}
=  - \sin λ
\frac{\partial R}{\partial x}
+  \cos λ
\frac{\partial R}{\partial y} \\
\frac{d R}{d λ}
=  - \sin λ \vec{e_x}
+  \cos λ \vec{e_y}
\\
\text{At } λ = 0 \\
\frac{d R}{d λ} = \vec{e_y}
\\
\text{At } λ = \pi/4 \\
\frac{d R}{d λ} = - \frac{1}{\sqrt{2}} \vec{e_x} + \frac{1}{\sqrt{2}} \vec{e_y}
\\
\frac{d R}{d λ} = \vec{e_r}
\\
\\
\text{Using the polar coordinate: } R(r, θ )=(1, λ) \\
\frac{d R}{d λ} =
  \frac{d r}{d λ}
\frac{\partial R}{\partial r}
+  \frac{d θ}{d λ}
\frac{\partial R}{\partial θ}
= \frac{\partial R}{\partial λ} = \vec{e_θ}
\\
\frac{d R}{d λ} = - \frac{1}{\sqrt{2}} \vec{e_x} + \frac{1}{\sqrt{2}} \vec{e_y}
\\
\frac{d R}{d λ} = \vec{e_r}
\\
T^1_1 =
\sum_{j=1}^2 \frac{\partial a^1}{\partial b^j}  \frac{\partial b^j}{\partial a^1}
 =
\frac{\partial a^1}{\partial b^1}  \frac{\partial b^1}{\partial a^1}
+
\frac{\partial a^1}{\partial b^2}  \frac{\partial b^2}{\partial a^1}
=
\frac{\partial x}{\partial r}  \frac{\partial r}{\partial x}
+
\frac{\partial x}{\partial θ}  \frac{\partial θ}{\partial x}
= \frac{d x}{d x} = 1
\\
T^2_1 =
\sum_{j=1}^2 \frac{\partial a^2}{\partial b^j}  \frac{\partial b^j}{\partial a^1}
 =
\frac{\partial a^2}{\partial b^1}  \frac{\partial b^1}{\partial a^1}
+
\frac{\partial a^2}{\partial b^2}  \frac{\partial b^2}{\partial a^1}
=
\frac{\partial y}{\partial r}  \frac{\partial r}{\partial x}
+
\frac{\partial y}{\partial θ}  \frac{\partial θ}{\partial x}
= \frac{d y}{d x} = 0
\\
B =
\begin{pmatrix}
\frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} \\
\frac{\partial θ}{\partial x} & \frac{\partial θ}{\partial y} \\
\end{pmatrix} \\
\vec{v} = v^i \vec{e_i}\\
\frac{d \vec{R}}{d \lambda} = \frac{\partial \vec{R}}{\partial a_i}
\frac{d a_i}{d \lambda}
\\
\frac{d \vec{R}}{d \lambda} = \frac{\partial a_i}{\partial \lambda}
\vec{e_{i}}
\\

F =
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial θ} \\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial θ} \\
\end{pmatrix} = \text{Jacobian matrix } J \\
\text{for } r = \sqrt{2}, \quad θ = π/4\\
\vec{ e_{r}} =
\frac{1}{\sqrt{2}} \vec{ e_{x}} + \frac{1}{\sqrt{2}} \vec{ e_{y}} \\

\text{Inverse Jacobian matrix } J^{-1} = B \\
\vec{ e_{θ}} =
- \vec{ e_{x}} + \vec{ e_{y}} \\

\text{for } r = 1, \quad θ = 0\\
\vec{ e_{r}} =
 \vec{ e_{x}}  \\

\vec{ e_{θ}} =
\vec{ e_{y}} \\
\frac{\partial x}{\partial θ} \\
\frac{\partial y}{\partial θ} \\

\frac{\partial \vec{R}}{\partial b^i} =
\frac{\partial a^j}{\partial b^i}  \frac{\partial \vec{R}}{ \partial a^j} \\
\vec{ e_{b^i}} =
\frac{\partial a^j}{\partial b^i}  \vec{ e_{a^j}} \\
\\
\frac{\partial \vec{R}}{\partial x} = \lim_{Δ x \rightarrow 0} \frac{\vec{R}(x+ \frac{1}{2}, y) - \vec{R}(x, y) }{\frac{1}{2}} = \vec{e_x}\\
\frac{\partial \vec{R}}{\partial x} = \lim_{Δ x \rightarrow 0} \vec{R}(x+ \frac{1}{4}, y) - \vec{R}(x, y)  = \frac{1}{4} \vec{e_x}\\
$$

$$
\begin{align}
y = \sum_{i=1}^n c_i x^i \rightarrow c_i x^i\\

\text{arc length} & = \int \left|\left|
d \vec{R} \right| \right| \\
& =  
\int_\lambda \left|\left| \frac{d \vec{R}}{ d \lambda} \right|\right| d \lambda  \\
& = \int_\lambda \sqrt{\frac{d \vec{R}}{ d \lambda} \cdot \frac{d \vec{R}}{ d \lambda}} d \lambda \quad \text { or }\\
  & = \int_\lambda
\sqrt{\frac{d q^i}{d \lambda} \frac{d q^j}{d \lambda}
\left( \frac{ \partial \vec{R}}{ \partial q^i}  \cdot \frac{ \partial \vec{R}}  { \partial q^j} \right) d \lambda} \\
\text{arc length} & = \int_\lambda \left|\left| \frac{d \vec{R}}{ d \lambda} \right|\right| d \lambda
\\
\end{align}
$$


$$
\| \mathbf{\vec{u}} \| \| \mathbf{\vec{v}} \| \cos θ = u^i v^j g_{ij}
\\
B: V \times V \rightarrow ℝ \\
B'_{ij} = F^k_iF^l_jB_{kl} \\
B_{kl} = B^i_kB^j_lB'_{ij} \\
L: V \rightarrow W \\
L: V \rightarrow V \\
L = L^1_1 \vec{e_1} ⊗ ε^1 + L^1_2 \vec{e_1} ⊗ ε^2 +
L^2_1 \vec{e_2} ⊗ ε^1 + L^2_2 \vec{e_2} ⊗ε^2 \\
\vec{e_i} ⊗ ε^j \\
\{\vec{e_1} ⊗ ε^1, \vec{e_1} ⊗ ε^2, \vec{e_2} ⊗ ε^1, \vec{e_2} ⊗ ε^2  \}
$$

$$
\begin{align}
L &= L^i_j \vec{e_i} ⊗ ε^j  \quad \quad \vec{v} = v^k \vec{e_k} \\
\vec{w} &= L(\vec{v}) = L^i_j \vec{e_i} ⊗ ε^j(v^k \vec{e_k}) \\
&= L^i_j v^k \vec{e_i} ⊗ ε^j(\vec{e_k}) \\
&= L^i_j v^k \vec{e_i} ⊗  \delta^j_k \\
&= L^i_j v^j \vec{e_i} \\
L & = L^k_l \vec{e_k} ε^l \\
& = L^k_l (B^i_k \vec{e'_i})(F^l_j ε'^j) \\
& = ( B^i_k L^k_l F^l_j) \vec{e'_i} ε'^j \\
L'^i_j & = B^i_k L^k_l F^l_j \\
B & = B_{ij} (ε^i ⊗ ε^j) \\
s & = B(\vec{v}, \vec{w}) \\
& = B_{ij} (ε^i ⊗ ε^j) (v^k\vec{e_k}, w^l\vec{e_l}) \\
& = B_{ij} (ε^i ( v^k\vec{e_k}) ) ⊗ (ε^j (w^l\vec{e_l})) \\
& = B_{ij} v^k w^l (ε^i ( \vec{e_k}) ) ⊗ (ε^j (\vec{e_l})) \\
& = B_{ij} v^k w^l \delta^i_k \delta^j_l \\
& = B_{ij} v^i w^j \\

T^{ab}_{cde} \\
T & = T^{ab}_{cde}\vec{e_i}⊗\vec{e_j} ⊗ ε^k ⊗ ε^l ⊗ ε^m  \\
&= T^{ab}_{cde} (B^i_a\vec{e'_i}) ⊗ (B^j_b\vec{e'_j})⊗
(F_k^c ε'^k) ⊗(F_l^d ε'^l) ⊗ (F_m^e ε'^m) \\
& = (B^i_a B^j_b  T^{ab}_{cde} F_k^c F_l^d F_m^e )\vec{e'_i} ⊗\vec{e'_j}⊗ ε'^k ⊗ε'^l ⊗ε'^m\\
T'^{ij}_{klm} & = B^i_a B^j_b  T^{ab}_{cde} F_k^c F_l^d F_m^e
\\
& A^{ab}_{cde} =  B^{ab}_{cde}\\
& A^{ab}_{cde} -  B^{ab}_{cde} = 0\\
& g^{ki} g_{ij}  = \delta^k_j \\
v_i & = g_{ij}v^j \\
g^{ki}v_i & = g^{ki}g_{ij}v^j \\
&= \delta^k_j v^j \\
&= v^k \\

 g(\vec{v}, \_ ) &= v_i ε^i \\
 g(\vec{v}, \_ ) &= g_{ik} ε^i ⊗ ε^k(v^j \vec{e_j})\\
&= g_{ik} v^j  ε^i ⊗ ε^k(\vec{e_j})\\
&= g_{ik} v^j  ε^i ⊗ \delta^k_j \\
&= g_{ij}v^j ε^i\\
v_i & = g_{ij}v^j \\
v^i & = g^{ij} v_j \\

& Y \text {: new coordinate system} \\
& X \text {: old coordinate system} \\
& ∂Y^m \text {: changes in the m component in the new coordinates} \\
& ∂X^p \text {: changes in the p component in the old coordinates} \\


\end{align}
$$

$$
\begin{align}
ε'^i &= \sum_j T^i_j ε^j\\
ε'^i(\vec{e'_k}) &= \sum_j T^i_j ε^j (\vec{e'_k}) \\
δ^i_k & = \sum_j T^i_j ε^j \left( \sum_l F^l_k\vec{e_l} \right) \\
& = \sum_j \sum_l T^i_j  \left(  F^l_k \right) ε^j(\vec{e_l}) \\
& = \sum_j \sum_l T^i_j  \left(  F^j_k \right) δ^j_l \\
& = \sum_j T^i_j   F^j_k  \\
T^i_j & = B^i_j \quad \quad \text{since } δ^i_k = \sum_j B^i_j   F^j_k \\
ε'^i &= \sum_j B^i_j ε^j\\
ε^i &= \sum_j F^i_j ε'^j\\
\end{align}
$$

$$
\begin{align}
\beta & = \sum_i \beta_i ε^i \\
& = \sum_i \beta_i \sum_j  F^i_j ε'^j \\
& =  \sum_j  \left( \sum_i \beta_i F^i_j \right) ε'^j \\
\beta'_i & =   \sum_i \beta_i F^i_j \\
\beta_i & =   \sum_i \beta'_i B^i_j \\
\end{align}
$$

$$
\begin{align}
\vec{e_r} & = F^1_1 \vec{e_x} +  F^2_1 \vec{e_y} \\
\vec{e_θ} & = F^1_2 \vec{e_x} +  F^2_2 \vec{e_y} \\
\end{align}
$$

$$
\begin{align}
\vec{e'_1} & = F^1_1 \vec{e_1} +  F^2_1 \vec{e_2} \\
\vec{e'_2} & = F^1_2 \vec{e_1} +  F^2_2 \vec{e_2} \\
\vec{e_1} & = B^1_1 \vec{e'_1} +  B^2_1 \vec{e'_2} \\
\vec{e_2} & = B^1_2 \vec{e_1} +  B^2_2 \vec{e'_2} \\
\vec{e'_i} & = \sum_{j=1}^n F^j_i \vec{e_j}  \\
\vec{e_k} & = \sum_{k=1}^n B^l_k \vec{e_l}  \\
\vec{e'_k} & = F^j_k \vec{e_j}  \\
\vec{e_j} & =  B^i_j \vec{e_i}  \\
v'^i & = B^i_j v^j \quad \quad v^j = F^j_i v'^i \quad \text { (contravariant components)}\\
v'_k & = F^j_k v_j \quad \quad v_j = B^k_j v'_k \quad \text { (covariant components)}\\  
& B^i_j F^j_k  = \delta^i_k\\

\vec{e_i} & = \sum_k \left( \sum_j B^i_j F^j_k \right) \vec{e_k} \\
\end{align}\\
\begin{pmatrix}
1 & 0 & \cdots & 0 \\
0 & 1& \cdots & 0 \\
\vdots  & \vdots  & \ddots & \vdots  \\
0 & 0 & \cdots & 1
\end{pmatrix} = δ^i_k \\
    δ^i_{k} =
    \begin{cases}
      1, & \text{if}\ i = k \\
      0, & \text{if}\ i ≠ k
    \end{cases}
$$

$$
\begin{align}
\vec{v} & = \sum_j v^j \vec{e_j} = \sum_j v^j \left( \sum_i B^i_j \vec{e'_i} \right)
= \sum_i \left(   \sum_j B^i_j v^j  \right) \vec{e'_i} \\
v'^i & =  \sum_j B^i_j v^j \\
\vec{v} & = \sum_j v'^j \vec{e'_j} = \sum_j v'^j \left( \sum_i F^i_j \vec{e_i} \right)
= \sum_i \left(   \sum_j F^i_j v'^j  \right) \vec{e_i} \\
v^i & =  \sum_j F^i_j v'^j \\

\end{align}
$$

$$
v'^i = \sum_j B^i_j v^j \\
v^i = \sum_j F^i_j v'^j \\
$$

$$
\begin{align}
\| \mathbf{\vec{v}} \|^2 & = \vec{v} \cdot \vec{v} \\
& = (v^1 \vec{e_1} + v^2 \vec{e_2}) \cdot (v^1 \vec{e_1} + v^2 \vec{e_2}) \\
& = (v^1)^2 (\vec{e_1} \cdot \vec{e_1}) + 2 v^1v^2 (\vec{e_1} \cdot \vec{e_2}) + (v^2)^2 (\vec{e_2} \cdot \vec{e_2}) \\
\| \mathbf{\vec{v}} \|^2 & = (v^1)^2 + (v^2)^2 \\
& \vec{e_i} \cdot \vec{e_j} \\
& \quad \quad \| \mathbf{\vec{v}} \|^2  = g_{μν}v^μv^ν \\
& \quad \quad g_{μν} = g_{νμ} \\
\end{align}
$$

$$
\| \mathbf{\vec{v}} \|^2  =
\begin{equation*}
\begin{pmatrix}
v^{1}  \\
v^{2} \\
v^{3} \\
v^{4}
\end{pmatrix}^T
\begin{pmatrix}
g_{11} & g_{12} & g_{13} & g_{14} \\
g_{21} & g_{22} & g_{23} & g_{24} \\
g_{31} & g_{32} & g_{33} & g_{34} \\
g_{41} & g_{42} & g_{43} & g_{44}
\end{pmatrix}
\begin{pmatrix}
v^{1}  \\
v^{2} \\
v^{3} \\
v^{4}
\end{pmatrix}
\end{equation*}
$$

$$
\begin{equation*}
g_{μν} \quad \text { for }  μ, ν ∈ \{1, 2, 3, 4 \} \\
g_{μν} =
\begin{pmatrix}
g_{11} & g_{12} & g_{13} & g_{14} \\
g_{21} & g_{22} & g_{23} & g_{24} \\
g_{31} & g_{32} & g_{33} & g_{34} \\
g_{41} & g_{42} & g_{43} & g_{44}
\end{pmatrix}
\end{equation*}
$$

$$
\begin{equation*}
L^a_{b} \quad \text { for }  a, b ∈ \{1, 2, 3 \} \\
L^a_{b} =
\begin{pmatrix}
L^1_{1} & L^1_{2} & L^1_{3}  \\
L^2_{1} & L^2_{2} & L^2_{3}  \\
L^3_{1} & L^3_{2} & L^3_{3}  \\
\end{pmatrix}
\end{equation*}
$$

$$
T^{a}_{bcd} \quad \text { for }  a, b, c, d ∈ \{1, 2, 3, 4 \}
$$

$$
T^{a}_{bcd} = A^{a}_{bcd} + B^{a}_{bcd}  \\
\\
T^1_{111} = A^1_{111} + B^1_{111} \\
T^1_{112} = A^1_{112} + B^1_{112} \\
\cdots \\
T^4_{444} = A^4_{444} + B^4_{444} \\

\text{If all components equals 0 for one coordinate system,} \\
\text{all components equal to 0 for all coordinate systems.}
\\
T^{ab}_{cde} = 0
$$

$$
\text { for }  μ, ν ∈ \{1, 2, 3, 4 \} \\
g_{μν}u^μ w^ν =
\begin{equation*}
\begin{pmatrix}
u^{1}  & u^{2} & u^{3} & u^{4} \\
\end{pmatrix}
\begin{pmatrix}
g_{11} & g_{12} & g_{13} & g_{14} \\
g_{21} & g_{22} & g_{23} & g_{24} \\
g_{31} & g_{32} & g_{33} & g_{34} \\
g_{41} & g_{42} & g_{43} & g_{44}
\end{pmatrix}
\begin{pmatrix}
w^{1}  \\
w^{2} \\
w^{3} \\
w^{4}
\end{pmatrix}
\end{equation*}
\begin{pmatrix}
2 & -1 & 0.5  \\
0 & 3.3 & 1  \\
1.2 & 0 & -1.2  \\
\end{pmatrix} \\
M = a \begin{pmatrix}
1 & 0  \\
0 & 0  \\
\end{pmatrix} + b
\begin{pmatrix}
0 & 1  \\
0 & 0  \\
\end{pmatrix} + c
\begin{pmatrix}
0 & 0  \\
1 & 0  \\
\end{pmatrix} + d
\begin{pmatrix}
0 & 0  \\
0 & 1  \\
\end{pmatrix} \\
\begin{pmatrix}
1   \\
0   \\
\end{pmatrix} ⊗
\begin{pmatrix}
1 & 0  \\
\end{pmatrix} =
\begin{pmatrix}
1 & 0  \\
0 & 0  \\
\end{pmatrix}\\
\begin{pmatrix}
1   \\
0   \\
\end{pmatrix} ⊗
\begin{pmatrix}
0 & 1  \\
\end{pmatrix} =
\begin{pmatrix}
0 & 1  \\
0 & 0  \\
\end{pmatrix}\\
\begin{pmatrix}
0   \\
1    \\
\end{pmatrix} ⊗
\begin{pmatrix}
1 & 0  \\
\end{pmatrix} =
\begin{pmatrix}
0 & 0  \\
1 & 0  \\
\end{pmatrix}\\
\begin{pmatrix}
0   \\
1   \\
\end{pmatrix} ⊗
\begin{pmatrix}
0 & 1  \\
\end{pmatrix} =
\begin{pmatrix}
0 & 0  \\
0 & 1  \\
\end{pmatrix}\\



$$

$$
\| \mathbf{\vec{v}} \|^2  =
\begin{equation*}
\begin{pmatrix}
v^{1}  & v^{2} & v^{3} & v^{4} \\
\end{pmatrix}
\begin{pmatrix}
g_{11} & g_{12} & g_{13} & g_{14} \\
g_{21} & g_{22} & g_{23} & g_{24} \\
g_{31} & g_{32} & g_{33} & g_{34} \\
g_{41} & g_{42} & g_{43} & g_{44}
\end{pmatrix}
\begin{pmatrix}
v^{1}  \\
v^{2} \\
v^{3} \\
v^{4}
\end{pmatrix}
\end{equation*}
$$
$$
T^{ab}_{bc} = \sum_b T^{ab}_{bc} = T^{a1}_{1c} + T^{a2}_{2c} + \cdots + T^{an}_{nc} = U^a_c \\
U^2_3 = T^{21}_{13} + T^{22}_{23} + \cdots + T^{2n}_{n3}  \\

g_{ij} v^k u^l δ^i_k δ^j_l = g_{ij} v^i u^j \\
$$

$$
L'^i_j = B^i_kL^k_lF^l_j \\
L^i_j = F^i_kL'^k_lB^l_j \\
$$

$$
\text{vector length} =
\begin{pmatrix}
a \\  b \\
\end{pmatrix}
 \rightarrow
\begin{pmatrix}
a &  b \\
\end{pmatrix}
\\
\text{vector length} = v_i v^i \\

 v_i \ne v^i \\

 v_i = g_{ij}v^j

$$

$$
\begin{align}
\frac{\partial^2 u}{\partial x^2} \\
d f  &= \frac{\partial f}{\partial q^i} d q^i \\
\frac{d f}{d \lambda} &= \frac{\partial f}{\partial q^i} \frac{d q^i}{d \lambda} \\
f(x, y) & = x^2 + 2 xy + 1 \\
x & = 2 \lambda \\
y & = \lambda^2 - 1 \\

\end{align}
$$

$$
total = (p^1 ε_1 + p^2 ε_2)(q_1 e^1 + q_2 e^2 ) \\
𝜙(\vec{v} + \vec{w}) =  𝜙(\vec{v} ) + 𝜙(\vec{w})\\
𝜙(\alpha \vec{v}) =  \alpha 𝜙(\vec{v} ) \\
\vec{v} ⊗ 𝜙 = (v^i \vec{e_i}) ⊗ (𝜙_j ε^j) = v^i 𝜙_j \vec{e_i} ⊗ ε^j \\
ψ ⊗ 𝜙 = (ψ_i ε^i) ⊗ (𝜙_j ε^j) = ψ_i 𝜙_j ε^i ⊗ ε^j \\

$$
$$
\vec{u}  =
\begin{equation*}
\begin{pmatrix}
1 & 4  \\
3 & -1  \\
\end{pmatrix}
\begin{pmatrix}
v^{1}  \\
v^{2} \\
\end{pmatrix}
\end{equation*}
\quad \quad u^i = \sum_j L^i_j v^j
$$

$$
\begin{equation}
    \vec{e_i} \cdot \vec{e_j} =
    δ_{ij} =
    \begin{cases}
      1, & \text{if}\ i = j \\
      0, & \text{otherwise}
    \end{cases}
  \end{equation}
$$

$$
\text{Dual vector space: {V*, F, + , ・}}\\
f_1, f_2  ∈ V^*, f_1 + f_2  ∈ V^*, a f_1 ∈ V^* \\
(f_1 + f_2) (x) = f_1(x) + f_2(x) \\
(af_1) (x) = af_1(x) \\
$$
$$
\text{Spacetime 4-vector} \\
\vec{e_i}
$$

$$
\begin{equation*}
\begin{pmatrix}
ct  \\
x  \\
y   \\
z  \\
\end{pmatrix}
\end{equation*} \\
$$

$$
\begin{equation*}
\begin{pmatrix}
ct_2  \\
x_2  \\
y_2   \\
z_2  \\
\end{pmatrix} +
\begin{pmatrix}
ct_1  \\
x_1  \\
y_1   \\
z_2  \\
\end{pmatrix} =
\begin{pmatrix}
ct_2 + ct_1  \\
x_2 + x_1 \\
y_2 + y_1   \\
z_2 + z_1  \\
\end{pmatrix}
\end{equation*}
$$

$$
\begin{equation*}
a
\begin{pmatrix}
ct_1  \\
x_1  \\
y_1   \\
z_2  \\
\end{pmatrix} =
\begin{pmatrix}
a ct_1  \\
a x_1 \\
a y_1   \\
a z_1  \\
\end{pmatrix}
\end{equation*}
$$

$$
\begin{equation*}
\begin{pmatrix}
0.4 & 0.5  \\
\end{pmatrix}

\begin{pmatrix}
11.5  \\
5 \\
\end{pmatrix}
\end{equation*}
$$

$$
ε^{i} (\vec{e_{j}}) =  δ^i_j
$$

$$
ε^{i} (\vec{e_{j}}) =  δ^i_j  \quad \quad \text { where }  
\begin{equation}
    δ^i_j =
    \begin{cases}
      1, & \text{if}\ i = j \\
      0, & \text{otherwise}
    \end{cases}
  \end{equation}
\\
ε^{y} (\vec{e_{r}}) =  ε^{r} (\vec{e_{y}}) = 0 \\
$$

$$
ε^{y} (\vec{e_{y}}) =  ε^{r} (\vec{e_{r}}) = 1 \\
ε^{y} (\vec{e_{r}}) =  ε^{r} (\vec{e_{y}}) = 0 \\
\text {apply covector } \vec{ε_{y}} \text {, a function, to vector } \\
\begin{align}
p &= 0.4 ε^{y} + 0.5  ε^{r} \\
p(\vec{e_{y}}) & = 0.4 ε^{y}(\vec{e_{y}}) + 0.5  ε^{r}(\vec{e_{y}}) \\
p(\vec{e_{y}}) & = 0.4 \\
p(\vec{e_{r}}) & = 0.5 \\
& \text {for } p = p_1 ε^{1} + p_2 ε^{2} \\
& \text{the component } p_i \text { for }  ε^{i} \text { equals }
p(\vec{e_{i}})  \\
& p(\vec{e_{1}}) = (p_1 ε^{1} + p_2 ε^{2} ) \vec{e_{1}} = p_1
\end{align} \\

$$

$$
p(q) = 0.4 q^1 + 0.5 q^2 ∈ V^*\\
(0.4 ε^{y}  + 0.5 ε^{r}) (11.5 \vec{e_{y}} + 5 \vec{e_{r}}) \\
$$

$$
\begin{align}
p(\vec{q}) &= (0.4 ε^{y}  + 0.5 ε^{r}) (11.5 \vec{e_{y}} + 5 \vec{e_{r}}) \\

 & = 0.4 \times 11.5 ε^{y}(\vec{e_{y}}) + 0.4 \times 5  ε^{y}(\vec{e_{r}})  \\
& \quad +  0.5 \times 11.5 ε^{r}(\vec{e_{y}}) + 0.5 \times 5  ε^{r}(\vec{e_{r}}) \\
& = 0.4 \times 11.5 ε^{y}(\vec{e_{y}})  + 0.5 \times 5  ε^{r}(\vec{e_{r}}) \\
& = 0.4 \times 11.5 + 0.5 \times 5 \\
\vec{p} & = 0.4 \vec{ε^{y}} + 0.5  \vec{ε^{r}} \\
\end{align}
$$

$$
\begin{align}
p & = 0.4 \vec{ε^{y}} + 0.5  \vec{ε^{r}} \\
p & = p(\vec{e'_1)} ε^{1} + p(\vec{e'_2)} ε^{2}  \\
p & = p(2 \vec{e_y} + \vec{e_r}) ε^{1} + p(3 \vec{e_y} + 2 \vec{e_r}) ε^{2}  \\
p & = (2p(\vec{e_y}) + p(\vec{e_r})) ε^{1} + (3p(\vec{e_y}) + 2p( \vec{e_r})) ε^{2}  \\
p & = (2 \times 0.4 + 0.5) ε^{1} + (3 \times 0.4 + 2 \times 0.5) ε^{2} \\
p & = 1.3 ε^{1} + 2.2 ε^{2}  \\
& \text {costs } 0.4 \times 2  + 0.5 \times 1 = 1.3 \\
& \text {costs } 0.4 \times 3  + 0.5 \times 2 = 2.2 \\
\\

\vec{e_1'} & = 2 \vec{e_y} + \vec{e_r} \\
\vec{e_2'} & = 3 \vec{e_y} + 2 \vec{e_r} \\
\vec{p} & = 0.4 \vec{ε^{y}} + 0.5  \vec{ε^{r}} = 0.4 \vec{ε'^{1}} + 0.5  \vec{ε'^{2}} \\
= 2 \vec{e_r} - \frac{3 \pi}{4}  \vec{e_\theta} \\
\end{align}
$$

$$
v = \begin{equation*}
\begin{pmatrix}
v_{1}  \\
v_{2}  \\
\vdots   \\
v_{n}  \\
\end{pmatrix}
\end{equation*} \\
\vec{e_1},  \vec{e_1}, \cdots, \vec{e_n}\\
\vec{v} = -\sqrt{2} \vec{e_x} - \sqrt{2} \vec{e_y} = 2 \vec{e_r} - \frac{3 \pi}{4}  \vec{e_\theta} \\
$$


$$
\begin{equation*}
\begin{pmatrix}
-\sqrt{2}  \\
-\sqrt{2}
\end{pmatrix}
\end{equation*}
$$

$$
\begin{equation*}
\begin{pmatrix}
2  \\
- \frac{3 \pi}{4}
\end{pmatrix}
\end{equation*}
$$

$$
\vec{v} = v_1 \vec{e_1} + v_2 \vec{e_2} + \cdots + v_n \vec{e_n}
$$

$$
\text{ Rank 0 Tensor} \\
\text{ Scalar} \\
1.5 \\
$$


$$
\text{ Rank 1 Tensor} \\
\text{ Vector} \\
\begin{equation*}
\begin{pmatrix}
v_{1}  \\
v_{2}  \\
\vdots   \\
v_{n}  \\
\end{pmatrix}
\end{equation*} \\
$$

$$
\text{ Rank 1 Tensor} \\
\text{ Vector} \\
\begin{equation*}
\begin{pmatrix}
x_{1}  \\
x_{2}  \\
\vdots   \\
x_{n}  \\
\end{pmatrix}
\end{equation*} \\
$$

$$
\text{ Rank 2 Tensor} \\
\text{ 2-D Matrix} \\
\begin{equation*}
\begin{pmatrix}
m_{1,1} & m_{1,2} & \cdots & m_{1,n} \\
m_{2,1} & m_{2,2} & \cdots & m_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
m_{m,1} & m_{m,2} & \cdots & m_{m,n}
\end{pmatrix}
\end{equation*} \\
$$

$$
\begin{equation*}
\begin{pmatrix}
G_{11} & G_{12} & G_{13} & G_{14} \\
G_{21} & G_{22} & G_{23} & G_{24} \\
G_{31} & G_{32} & G_{33} & G_{34} \\
G_{41} & G_{42} & G_{43} & G_{44}
\end{pmatrix} = \dfrac{8\pi G}{c^4}
\begin{pmatrix}
T_{11} & T_{12} & T_{13} & T_{14} \\
T_{21} & T_{22} & T_{23} & T_{24} \\
T_{31} & T_{32} & T_{33} & T_{34} \\
T_{41} & T_{42} & T_{43} & T_{44}
\end{pmatrix}
\end{equation*} \\
$$
$$
T'^{abc...}_{uvw...} = \left( B^a_hB^b_iB^c_j \cdots \right)T^{hij...}_{pqr...} \left(  F^p_uF^q_vF^r_w \cdots \right) \\

T^{hij...}_{pqr...} = \left( F_a^hF_b^iF_c^j \cdots \right)T^{abc...}_{uvw...} \left(  B_p^uB_q^vB_r^w \cdots \right) \\
$$

$$
g(\vec{u}, \vec{v}) \rightarrow u^iv^jg_{ij} \\
$$

$$
g =
\begin{pmatrix}
\vec{e_1} \cdot \vec{e_1} & \vec{e_1} \cdot \vec{e_2} \\
\vec{e_2} \cdot \vec{e_1} & \vec{e_2} \cdot \vec{e_2} \\
\end{pmatrix}

$$

$$
\begin{equation*}
\begin{pmatrix}
g_{11} & g_{12} & \cdots & g_{1n} \\
g_{21} & g_{22} & \cdots & g_{2n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
g_{n1} & g_{n2} & \cdots & g_{nn}
\end{pmatrix}
\end{equation*}
\quad \quad  \text{where element } i, j \text { equals }
 (\vec{e_i} \cdot \vec{e_j}) \\
$$


$$
\begin{equation*}
\begin{pmatrix}
p_{1} & p_{2} \\
\end{pmatrix}
\begin{pmatrix}
q_{1} \\
q_{2} \\
\end{pmatrix}
\end{equation*} \\
lb
$$

$$
total =
\begin{equation*}
\begin{pmatrix}
p_{1} & p_{2} \\
\end{pmatrix}
\begin{pmatrix}
q^{1} \\
q^{2} \\
\end{pmatrix}
\end{equation*}  = \sum_i p_i q^j = p_i q^j
\\lb \\
total = p_i q^i \\
$$


$$
\begin{equation*}
\begin{pmatrix}
0.4 \times 5 & 0.5 \times 5 \\
\end{pmatrix}
\begin{pmatrix}
11.5 \div 5 \\
5 \div 5 \\
\end{pmatrix}
\end{equation*}
= 7.1
\\5 lb
$$
