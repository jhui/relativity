$$
\begin{align}

& Section 1 \\

A = V \Lambda V^{-1}, \\
\\
x_n &= A x_{n-1} = A^2 x_{n-2} = \cdots = A^n x_0. \\
\\
A^n & = V \Lambda V^{-1} (V\Lambda V^{-1}) \cdots V\Lambda V^{-1} \\
& = V \Lambda^n V^{-1}. \\

& \langle v_i, v_j \rangle = 0 \text{ if } i \ne j, \\
& \langle v_i, v_i \rangle = 1. \\
& \lambda \langle \boldsymbol x, \boldsymbol  y \rangle = \mu \langle \boldsymbol x, \boldsymbol  y \rangle. \\
\\
& \boldsymbol x = \sum^n_{i=1} \alpha_i \boldsymbol b_i, \\
& \boldsymbol y = \sum^n_{i=j} \beta_i \boldsymbol b_j. \\

\\
 \langle \boldsymbol x, \boldsymbol  y \rangle &=
\Bigg  \langle \sum^n_{i=1} \alpha_i \boldsymbol b_i,  \sum^n_{j=1} \beta_i \boldsymbol b_j \Bigg \rangle
\\
&= \sum^n_{i=1} \sum^n_{j=1} \alpha_i \langle \boldsymbol b_i, \boldsymbol b_j \rangle  \beta_j \\
&= \hat{\boldsymbol x}^\top \boldsymbol A \hat{\boldsymbol y}, \quad \\
\\
& \text{ where }  \hat{\boldsymbol x} =[\alpha_1, \cdots, \alpha_n], \hat{\boldsymbol y} =[\beta_1, \cdots, \beta_n] \text{ and }\\
& \quad \quad \quad \boldsymbol A_{ij} = \langle \boldsymbol b_i, \boldsymbol b_j \rangle . \\
\hat{\boldsymbol x}^\top \boldsymbol A \hat{\boldsymbol x} &=
 \langle \boldsymbol x, \boldsymbol  x \rangle = \| x \|^2 \ge 0, \\
 \langle \boldsymbol x, \boldsymbol  y \rangle &= \hat{\boldsymbol x}^\top \boldsymbol A \hat{\boldsymbol y}. \\

x^\top M x =
 \begin{pmatrix}
 x_1 & x_2 \\
 \end{pmatrix}
 \begin{pmatrix}
 2 & 6 \\
 6 & 18 \\
 \end{pmatrix}
 \begin{pmatrix}
 x_1  \\
 x_2 \\
 \end{pmatrix}
 & = 2x_1^2 + 12x_1 x_2 + 18 x_1x_2 = 2( x_1 + 3 x_2)^2 \ge 0. \\
 & \langle b_i, b_j \rangle = 0 \text{ if } i \ne j, \\
 & \langle b_i, b_i \rangle = 1. \\
 & \langle b - p, a \rangle = 0 \\
 \Rightarrow \quad & \langle b - \hat{x} a, a \rangle = 0 \\
\Rightarrow \quad  & \langle b, a \rangle -  \langle \hat{x} a, a  \rangle = 0 \\
\Rightarrow \quad & \langle b, a \rangle = \hat{x} \langle  a, a \rangle = \hat{x} \| a\|^2 \\
\Rightarrow \quad &   \hat{x} = \frac{ \langle b, a \rangle}{\| a\|^2 } = \frac{ b^\top a}{\| a\|^2 } = \frac{ a^\top b}{\| a\|^2 }.
\end{align}
$$

$$
\begin{align}
& section 2\\
\\
A &=
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{pmatrix}, \quad
B =
\begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
\end{pmatrix}.
\\
\\
AB &= \begin{pmatrix}
A B_{*1}  & A B_{*2} \\
\end{pmatrix}.  
\\ \\
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{pmatrix}
\begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
\end{pmatrix} & =
\begin{pmatrix}

\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{pmatrix}
\begin{pmatrix}
b_{11} \\ b_{21}  \\
\end{pmatrix}
&
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{pmatrix}
\begin{pmatrix}
b_{12} \\ b_{22}  \\
\end{pmatrix}
\end{pmatrix}
.
\\
\\
AB &= \begin{pmatrix}
A_{1*}  B \\ A_{2*}  B \\
\end{pmatrix}  \\
\\
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{pmatrix}
\begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
\end{pmatrix} & =\begin{pmatrix}

\begin{pmatrix}
a_{11} & a_{12} \\
\end{pmatrix}
\begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
\end{pmatrix}
\\
\\
\begin{pmatrix}
a_{21} & a_{22} \\
\end{pmatrix}
\begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
\end{pmatrix}

\end{pmatrix}
.\\
\boldsymbol T(\boldsymbol x) = \boldsymbol A \boldsymbol x.
\\
&\begin{bmatrix}
1 & 0 & 0 \\
-4 & 1 & 0 \\
-7 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 3 & 6 \\
4 & 5 & 6 & 8 \\
7 & 8 & 9 & 10 \\
\end{bmatrix} =
\begin{bmatrix}
1  \\
-4  \\
-7 \\
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 3 & 6 \\
\end{bmatrix} +
\begin{bmatrix}
0  \\
1  \\
0 \\
\end{bmatrix}
\begin{bmatrix}
4 & 5 & 6 & 8 \\
\end{bmatrix} +
\begin{bmatrix}
0  \\
0  \\
1 \\
\end{bmatrix}
\begin{bmatrix}
7 & 8 & 9 & 10 \\
\end{bmatrix},
\\
& \min \| b - A \hat{x} \|^2. \\
& \det(A) = \det(V \Lambda V^{-1}) = \det(V ) \det( \Lambda) \det( V^{-1})  = \det( \Lambda) = Tr(\Lambda). \\
f(x) & = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots,\\
&= \sum^{\infty}_{n=0} \frac{f^{(n)}(a)}{n!} (x-a)^n. \\
\\
e^x & = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!}+ \cdots.\\ \\
 & \quad \frac{d}{dx} x^n  = n x^{n-1} \\
 &  \quad  \frac{d}{dx} e^x = e^x \\
&  \quad \frac{d}{dx} \ln(x) = \frac{1}{x} \\
\text{Sum rule: } & \quad  (f+g)'  = f' + g' \\
\text{Product rule: } & \quad  (fg)'  = f'g + fg' \\
\text{Quotient rule: } & \quad (\frac{f}{g})'  = \frac{f'g - fg'}{g^2} \\
\text{Chain rule: } & \quad  (f(g(x)))' = g \circ f = g'(f(x))f'(x) \\
\text{Chain rule: } & \quad \frac{dy}{dx} = \frac{dy}{du}  \frac{du}{dx}. \\
\text{Chain rule: } & \quad \frac{d f}{dt} = \frac{\partial f}{\partial x_1}  \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}  \frac{\partial x_2}{\partial t}. \\
\\ \\
& \quad \int (\frac{1}{x}) dx = \ln(x) + C\\
& \quad \int e^x dx = e^x + C\\
& \quad \int \ln(x) dx = x\ln(x) -x + C\\
& \quad \int x^n dx = \frac{x^{n+1}}{n+1} + C\\
\text{Sum rule: } & \quad  \int (f+g)dx  = \int f dx + \int g dx \\
\text{Integration by part: } & \quad  \int udv  = uv - \int v du \\
\text{Substitution Rule: } & \quad  \int f(g(x))g'(x)dx  = \int f(u) du \quad \quad \text{where } u = g(x).\\
∇_x f & = \text{grad } f = \Big[
\frac{\partial f(x)}{\partial x_1} , \frac{\partial f(x)}{\partial x_2}
, \frac{\partial f(x)}{\partial x_3}, \cdots, \frac{\partial f(x)}{\partial x_n}
\Big]^\top.

\end{align}
$$

$$
\begin{align}
& Section 3\\

\| \theta x - y \|^2&  = (\theta x - y)^\top(\theta x-y) \\
∇_x & = 2(\theta x -b)^\top \theta. \\
\\
x^\top A x & = x^\top Q \Lambda Q^\top x \\
&= (Q^\top x)^\top \Lambda (Q^\top x) \\
 &= \sum^n_{i=1} \lambda_i (q_i^\top x)^2 \\
& \le \lambda_\max \| x \|^2, \quad \quad \text{as $q_i$ is a unit vector. } \\
f(x) & = \| A x - b \|^2 \\
∇_x f(x) & = 2 A^\top(Ax -b ) \\
∇^2_x f(x) & = 2 A^\top A \ge 0.  \\
& \min_x \quad \frac{1}{2} x^\top Q x + c^\top x \\
& \text{subject to} \quad Ax \le b.\\ \\
f(\lambda x_1 + (1-\lambda)x_2) & \le \lambda f(x_1) + (1-\lambda)f(x_2), \quad \text{for $\lambda\in[0, 1]$.} \\
\sum^k_{i=1} c_i v_i = 0 & \Rightarrow c = \boldsymbol 0 \text{ if $v_i$ are independent of each other.} \\
\text{span}(x_1, \cdots, x_k) & = \Bigg\{ \sum^k_{i=1} c_i x_i : c_i \in \mathbb{R} \Bigg\}. \\
U & = \{ x \in \mathbb{R}^n : Ax =b \}. \\
x^\top A x &  = \sum^n_{i=1} \lambda_i (q_i^\top x)^2. \\
\end{align}
$$

$$
\begin{align}
section 3b \\
&\min_x f(x), & &x =(x_1, \cdots, x_n) \\
& \text{subject to } &
 g_i(x) \le 0, \quad \quad  \quad  & i \in \{ 1, \cdots, m \},  \quad  \quad \\
& & h_j(x) = 0, \quad  \quad \quad  &  j \in \{ 1, \cdots, p \}.  \quad  \quad\\

f\Big(

\Big)

\end{align}
$$


$$
\begin{align}
& section 4 \\ \\
f\Big(
  \begin{bmatrix}
  x_1  \\
  x_2  \\
  \end{bmatrix}
\Big) & = \frac{1}{2}
\begin{bmatrix}
x_1  \\
x_2  \\
\end{bmatrix} ^\top
\begin{bmatrix}
a & b \\
b & c  \\
\end{bmatrix}
\begin{bmatrix}
x_1  \\
x_2  \\
\end{bmatrix} +
\begin{bmatrix}
d  \\
e  \\
\end{bmatrix} ^\top
\begin{bmatrix}
x_1  \\
x_2  \\
\end{bmatrix} \\
∇ f\Big(
  \begin{bmatrix}
  x_1  \\
  x_2  \\
  \end{bmatrix}
\Big) & = \begin{bmatrix}
x_1  \\
x_2  \\
\end{bmatrix} ^\top
\begin{bmatrix}
a & b \\
b & c  \\
\end{bmatrix} + \begin{bmatrix}
d  \\
e  \\
\end{bmatrix} ^\top.
\\
\\
J(x)  & = f(x) + \sum^m_{i=1} \boldsymbol 1 (g_i(x)) \\
& \text{where }
 \boldsymbol 1(z) =
\begin{cases}
  0 \text{ if } z \le 0, \\
  \infty \text{ if } z \gt 0.
\end{cases}
\\
\mathcal{L}(\boldsymbol x, \boldsymbol \lambda)  & = f(\boldsymbol x) + \sum^m_{i=1} \mu_i g_i(\boldsymbol x). \\
& \max_{\mu_i} \mu_i g_i(\boldsymbol x) \\  
\\
\min_{\boldsymbol x} \max_{\boldsymbol \lambda, \boldsymbol \mu} \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda , \boldsymbol \mu)  & = \min_{\boldsymbol x} \max_{\boldsymbol \lambda, \boldsymbol \mu} \Bigg( f(\boldsymbol x)  
+ \sum^p_{j=1} \lambda_i h_i(\boldsymbol x) + \sum^m_{i=1} \mu_i g_i(\boldsymbol x) \Bigg), \\
& \quad \text{ such that }  \mu_i \ge 0 \quad \text{ for } i = 1, \cdots, m.
\\
\\
\\
 \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda , \boldsymbol \mu)  & =  f(\boldsymbol x)  
+ \sum^p_{j=1} \lambda_j h_j(\boldsymbol x) + \sum^m_{i=1} \mu_i g_i(\boldsymbol x), \quad \text{ with }  \mu_i \ge 0  \text{ for } i = 1, \cdots, m.
\\
\\
\min_{\boldsymbol x} \max_{\boldsymbol \lambda} \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda ) \quad &? = \quad  \max_{\boldsymbol \lambda} \min_{\boldsymbol x} \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda). \\
\min_x l(x, y) & \le \max_y l(x,y), \quad \quad \text{ for } \forall x, \forall y. \\
\\
& \max_y \min_x l(x, y)  \le \min_x \max_y l(x,y). \\
\inf f(x) & \le f(x) \le \sup f(x).\\ \\
f^* & = \inf_{\boldsymbol x} \text{ } \sup_{\boldsymbol \mu \ge 0, \boldsymbol \lambda} \mathcal{L}(\boldsymbol x, \boldsymbol\lambda, \boldsymbol \mu).\\
l(\boldsymbol \lambda, \boldsymbol \mu ) & = \inf_{\boldsymbol x}  \mathcal{L}(\boldsymbol x, \boldsymbol\lambda, \boldsymbol \mu). \\
\\

\max_{\boldsymbol \lambda, \boldsymbol \mu} &   \quad l(\boldsymbol \lambda, \boldsymbol \mu ), \quad \text{subject to } \boldsymbol \mu > 0. \\
\\
l^* & = \sup_{\boldsymbol \mu \ge 0, \boldsymbol \lambda}
\inf_{\boldsymbol x} \mathcal{L}(\boldsymbol x, \boldsymbol\lambda, \boldsymbol \mu). \\

l^* & \le f^*. \\

h_j(x) = 0 \quad &  \Rightarrow \quad h_j(x) \le 0 \text{ and } - h_j(x) \le 0.


\end{align}
$$


$$
\begin{align}
& section 5 \\
\\
& \min f(x) \\
& \text{ subject to } \\
& \quad \quad g_i(x) \le 0, \quad \text{for } i = 1, \cdots, m, \\
& \quad \quad h_j(x) = 0, \quad \text{for } j = 1, \cdots, p.
\\ \\
\exists x, \quad &  g_i(x) <0 \quad \text{for } i = 1, \cdots, m, \text{ and, }\\
& h_j(x)=0 \quad \text{for } i = 1, \cdots, p. \\
& \min f(x, y)  = x + y \\
& \text{subject to } x^2 + y^2 = 25.\\
& \mu_i g_i(x) = 0. \\
& a_1 x_1 + \cdots + a_k x_k, \quad with \sum^k_{i=1} a_i = 1 \text{ and } a_i \ge 0. \\
\text{Stationarity:} & \\
& - ∇f(x) = \sum^m_{i=1} \mu_i ∇g_i(x) + \sum^p_{j=1} \lambda_j ∇h_i(x). \\
\text{Primal feasibility:} & \\
& g_i(x) \le 0, \quad \text{for } i = 1, \cdots, m, \\
& h_j(x) = 0, \quad \text{for } j = 1, \cdots, p. \\
\text{Dual feasibility:} & \\
& \mu_i \ge 0 , \quad \text{for } i = 1, \cdots, m.\\
\text{Complementary slackness:} & \\
& \mu_i g_i(x) = 0 , \quad \text{for } i = 1, \cdots, m. \\
\\
& \textbf{Primal problem:}  \\
& \min_x f(x) \\
& \text{ subject to } \\
& \quad \quad g_i(x) \le 0, \quad \text{for } i = 1, \cdots, m, \\
& \quad \quad h_j(x) = 0, \quad \text{for } j = 1, \cdots, p.
\\ \\
& \textbf{Dual problem:}  \\

& \max_{ \boldsymbol \lambda , \boldsymbol \mu}  \text{ } \min_{\boldsymbol x} \quad f(\boldsymbol x)  
+ \sum^p_{j=1} \lambda_j h_j(\boldsymbol x) + \sum^m_{i=1} \mu_i g_i(\boldsymbol x),\\
 &\text{ subject to }  \boldsymbol \mu \ge 0.
\\
\\
& \text{ or } \\ \\
& \max_{ \boldsymbol \lambda , \boldsymbol \mu} l( \boldsymbol \lambda , \boldsymbol \mu) \quad \text{ subject to }  \boldsymbol \mu \ge 0. \\
& \text{ where} \\
& \quad \quad l( \boldsymbol \lambda , \boldsymbol \mu) = \min_x  \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda , \boldsymbol \mu) \\  
& \quad \quad \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda , \boldsymbol \mu) = f(\boldsymbol x)  
+ \sum^p_{j=1} \lambda_j h_j(\boldsymbol x) + \sum^m_{i=1} \mu_i g_i(\boldsymbol x). \\

& \min_{x \in \mathbb{R}^d} \quad c^\top x, \\
& \text{ subject to } A x \le b, \quad \text{where } A \in \mathbb{R}^{m \times d} \text{ and } b \in \mathbb{R}^{m}.
\\
\\
& \max_{ \boldsymbol \lambda , \boldsymbol \mu} l( \boldsymbol \lambda , \boldsymbol \mu) \quad \text{ subject to }  \boldsymbol \mu \ge 0. \\
& \text{ where} \\
& \quad \quad l( \boldsymbol \lambda , \boldsymbol \mu) = \min_x  \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda , \boldsymbol \mu) \\  
& \quad \quad \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda) = \boldsymbol c\top \boldsymbol x
+  \boldsymbol \lambda^\top (\boldsymbol {Ax} - \boldsymbol b)
= (\boldsymbol c + \boldsymbol{A}^\top \boldsymbol \lambda)^\top \boldsymbol x - \boldsymbol \lambda^\top \boldsymbol b
. \\
\\
& \boldsymbol c
+ \boldsymbol {A}^\top \boldsymbol \lambda. \\
l(\boldsymbol \lambda)& = \inf_x  \mathcal{L}(\boldsymbol x,  \boldsymbol \lambda) = \inf_x \boldsymbol \quad (c + \boldsymbol{A}^\top \boldsymbol \lambda)^\top \boldsymbol x - \boldsymbol \lambda^\top \boldsymbol b = 0 x - \boldsymbol \lambda^\top \boldsymbol b = - \boldsymbol \lambda^\top \boldsymbol b. \\
\\
&\max_{ \boldsymbol \lambda \in \mathbb{R}^m} \quad \quad \quad - \boldsymbol \lambda^\top \boldsymbol b\\
& \text{ subject to }  \quad \boldsymbol c +  \boldsymbol A^\top  \boldsymbol \lambda = 0 \text{ and }  \boldsymbol \lambda \ge 0.  \\
\\
&\min_{ \boldsymbol x} \quad \quad \quad \frac{1}{2} \boldsymbol x^\top  \boldsymbol{Qx} +  \boldsymbol c^\top x\\
& \text{ subject to }  \quad \boldsymbol {Ax} \le  \boldsymbol b.  \\
\\
\mathcal{L}(\boldsymbol x, \boldsymbol \lambda) &= \frac{1}{2}
\boldsymbol x^\top  \boldsymbol{Qx} +  \boldsymbol c^\top x
+ \boldsymbol \lambda^\top ( \boldsymbol {Ax} - \boldsymbol {b}) \\
&= \frac{1}{2}
\boldsymbol x^\top  \boldsymbol{Qx} +( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda )^\top \boldsymbol x - \boldsymbol \lambda^\top \boldsymbol b, \\
∇_x \mathcal{L}(\boldsymbol x, \boldsymbol \lambda) &=
 \boldsymbol{Qx} +( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda )^\top. \\
 \boldsymbol{x} & = -  \boldsymbol Q^{-1}( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda ). \\
 \\
 l(\boldsymbol \lambda ) & = \inf_{x} \mathcal{L}(\boldsymbol x, \boldsymbol \lambda) =
 - \frac{1}{2} ( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda )^\top \boldsymbol Q^{-1} ( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda ) -
 \boldsymbol \lambda^\top \boldsymbol b.
\\
\\
& \max_{\boldsymbol \lambda }  - \frac{1}{2} ( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda )^\top \boldsymbol Q^{-1} ( \boldsymbol c + \boldsymbol A^\top \boldsymbol \lambda ) -
\boldsymbol \lambda^\top \boldsymbol b, \\
& \text{ subject to } \boldsymbol \lambda \ge 0.
\\
\\
I_C(\boldsymbol x) &= \begin{cases}
                0 \quad \quad  \boldsymbol x \in x, \\
                \infty \quad \quad \boldsymbol x \not\in x.
            \end{cases}
\\
f(x) &= \min_{y \in C} g(x, y). \\
& \min_{\beta} \quad \| \boldsymbol  y - \boldsymbol {X \beta} \|^2_2 \\
& \text{ subject to } \| \boldsymbol {\beta} \|_1 \le s. \\ \\
\mathcal{L}(\boldsymbol {\beta}, \boldsymbol \lambda) & = \| \boldsymbol  y - \boldsymbol {X \beta} \|^2_2 + \boldsymbol \lambda \| \boldsymbol {\boldsymbol \beta} \|_1. \\ \\
& \min_{\boldsymbol \beta , \beta_0, \xi} \quad \frac{1}{2} \| \boldsymbol \beta \|^2_2 + C \sum_i^n \xi_i, \\
& \text{ subject to } \\
& \quad \xi_i \ge 0, \\
& \quad y_i(x^T \beta + \beta_0) \ge 1 - \xi_i. \\
\\
\boldsymbol x^* & = \arg \min_{\boldsymbol x} \mathcal{L}(\boldsymbol x,   \mu). \\
& 1. \text{ Find } \boldsymbol x^* = \arg \min_{\boldsymbol x} \mathcal{L}(\boldsymbol x,   \mu) \\
& 2. \text{ Compute } \frac{d l}{ d \mu} = \frac{d \mathcal{L}}{ d \mu} (\boldsymbol x^*,  \mu) \\
& 3.  \text{ } \mu =  \mu + \alpha \frac{d l}{ d \mu}. \\
& \text{ Given $\lambda$, find } \boldsymbol x^* = \arg \min_{\boldsymbol x} \mathcal{L}(\boldsymbol x,   \lambda)\\
& \mathcal{L} = f(x) + \mu g(x) \\

\end{align}

$$


$$
\begin{align}
\\
& section 6 \\
\\
\max_{\boldsymbol w, b, r} \quad & r \\
\text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b\Big) \ge r, \quad \text{for } i = 1, \cdots, n, \quad r > 0,\\
& x_1 = x_1' + r \frac{w}{\| w \| }. \\
\Big\langle  \boldsymbol {w}, \boldsymbol x_1' \Big\rangle + b & = 0 \\
\Big\langle  \boldsymbol {w}, \boldsymbol x_1 - r \frac{w}{\| w \|} \Big\rangle + b & = 0 \\
\Big\langle  \boldsymbol {w}, \boldsymbol x_1 \Big\rangle + b - r \frac{\langle  \boldsymbol {w},  \boldsymbol {w}  \rangle}{\| w \|} & = 0 \\
 0 -  r \frac{\| w \|^2 }{\| w \|} & = 0\\
 r & = \frac{1}{\| w \|}. \\
 \\
 \max_{\boldsymbol w, b, r} \quad & r \\
 \text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b\Big) \ge r, \| \boldsymbol w \| = 1, r > 0. \\ \\
 \min_{\boldsymbol w, b} \quad & \frac{1}{2} \| w\|^2 \\
 \text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b\Big) \ge 1, \\ \\
& y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b\Big) \ge 1, \\
& \Rightarrow
y_i \Bigg(
\Big\langle  \frac{\boldsymbol {w'}}{ \| \boldsymbol w' \| }, \boldsymbol x_i \Big\rangle + b
  \Bigg) \ge r \quad \text{since $w$ is a unit vector.}\\
  & \Rightarrow

  y_i \Bigg(
  \Big\langle  \frac{\boldsymbol {w'}}{ \| \boldsymbol w' \| r}, \boldsymbol x_i \Big\rangle + \frac{b}{r}
    \Bigg) \ge 1 \\
& \Rightarrow
y_i \Bigg(
\Big\langle  \boldsymbol {w''}, \boldsymbol x_i \Big\rangle + {b''}
  \Bigg) \ge 1 ,\quad \text{assume } \boldsymbol {w''} = \frac{\boldsymbol {w'}}{\| \boldsymbol {w'} \| r }, b'' = \frac{b}{r}. \\
  \\
  \max_{\boldsymbol w, b, r} \quad & r \quad \text{ or } \quad \max_{\boldsymbol w, \boldsymbol w'', b''} \quad \frac{1}{\| \boldsymbol w \|} \quad \text{ or } \quad \max_{\boldsymbol w'', b''} \quad \frac{1}{\| \boldsymbol w'' \|} \\
  \text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w''}, \boldsymbol x_i \rangle + b'' \Big) \ge 1.  \\ \\
   \max_{\boldsymbol w, b} \quad & \frac{1}{\| \boldsymbol w \|} \\
  \text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b \Big) \ge 1.  \\ \\
  \min_{\boldsymbol w, b} \quad & \frac{1}{2}{\| \boldsymbol w \|}^2 \\
 \text{subject to} \quad & y_i \Big(\langle  \boldsymbol {w}, \boldsymbol x_i \rangle + b \Big) \ge 1  \\ \\

 \min_{\boldsymbol w , b, \xi} \quad & \frac{1}{2} \| \boldsymbol w \|^2_2 + C \sum_i^n \xi_i, \\
 \text{ subject to } & \\
& \quad \xi_i \ge 0, \\
& \quad y_i(\langle \boldsymbol w, \boldsymbol x \rangle + b) \ge 1 - \xi_i. \\

\end{align}

$$
