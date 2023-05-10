$$
\begin{align}

& Section 1 \\
& L: X \rightarrow \mathbb{F}, \quad \mathbb{F} \in \{ \mathbb{R}, \mathbb{C}\}, \text{ there is an exactly one } g \in X, \\
& Lf = \langle f, g \rangle_X, \quad \text{ for } \forall f \in X. \\
& f(\lambda x ) = \lambda f(x), \\
& f(x + y) = f(x) + f(y). \\
& \text{linearity } + \text{bijective} + \| f(x) \|_{Y} =  \| x \|_{X}. \\
\\
& \text{linearity } + \text{bijective} + \langle u, v \rangle_X =  \langle Mu, Mv \rangle_Y. \\
& \forall n ≥ 1, \forall (a_1, a_2, \cdots, a_n) \in \mathbb{R}^n, \forall(x_1, x_2, \cdots , x_n) \in X^n, \\
& \sum^n_{i=1}\sum^n_{j=1} a_i a_j k(x_i, x_j ) \ge 0. \\

& x_1, x_2, x_3, \cdots, x_m. \\
& k(x_i, x_j ) =  k(x_j, x_i). \\
& \boldsymbol K = \begin{pmatrix}
k(x_1, x_1) & k(x_1, x_2) & \cdots & k(x_1, x_m) \\
k(x_2, x_1) & k(x_2, x_2) & \cdots & k(x_2, x_m) \\
\vdots \\
k(x_m, x_1) & k(x_m, x_2) & \cdots & k(x_1, x_m) \\  
\end{pmatrix}. \\
\\
& \phi(\boldsymbol x_i) = \Big(  
\sqrt{\lambda_1} v^{(i)}_1, \sqrt{\lambda_2} v^{(i)}_2, \cdots, \sqrt{\lambda_m} v^{(i)}_m
  \Big), \\
  & \phi(\boldsymbol x_j) = \Big(  
  \sqrt{\lambda_1} v^{(j)}_1, \sqrt{\lambda_2} v^{(j)}_2, \cdots, \sqrt{\lambda_m} v^{(j)}_m
    \Big), \\
& \text{where } v_1, v_2, \cdots, v_m \in \mathbb{R}^m. \\
\\
& \langle
\phi(\boldsymbol x_i),  \phi(\boldsymbol x_j)
\rangle_{\mathbb{R}^m}   
= \sum^m_{k=1} \lambda_1 v_k^{(i)} v_k^{(j)} = (\boldsymbol Q
\boldsymbol \Lambda \boldsymbol Q^\top )_{ij} = \boldsymbol K_{ij} = k(\boldsymbol x_i,\boldsymbol x_j). \\

\boldsymbol u^\top \boldsymbol K \boldsymbol u
& = \sum_{i,j =1}^m u_i u_j K_{ij} \\
&= \sum_{i,j =1}^m u_i u_j \langle \phi(\boldsymbol x_i ), \phi(\boldsymbol x_j ) \rangle \\
&= \Bigg\langle
\sum^m_{i=1} u_i \phi(\boldsymbol x_i), \sum^m_{j=1} u_j \phi(\boldsymbol x_j)
\Bigg\rangle \\
&= \Bigg\|
\sum^m_{i=1} u_i \phi(\boldsymbol x_i)
\Bigg\|^2 \ge 0. \\

& k(x_i, x_j): \mathbb{R}^m \times \mathbb{R}^m \rightarrow \mathbb{R}, \quad \text{where $m$ is the number of features in $x_i$ and $x_j$}, \\
\\
& X = \{x_1, x_2, \cdots, x_n \}. \\
\\
& \mathcal{H} \text{ be a Hilbert space, and }\\
&\text{$L$ be a continous linear functional on } \mathcal{H}. \\
\\
& Lf = \langle f , g \rangle_{\mathcal{H}}, \quad \quad \quad g \in  \mathcal{H}, \quad \forall f \in \mathcal{H}. \\


\end{align} \\
$$

$$
\begin{align} \\
& section 2 \\
& \| \delta \frac{x}{2 \|x\|_X}\|_x = \frac{\delta}{2} \\
\\
& \quad \text{As } ‖λx‖ = λ‖x‖ \text{ for } λ>0,
\\
\\
& \quad 1  \ge  \| Tv \| = \Big\| T\Big( \frac{\delta}{2} \frac{x}{\| x \|_X} \Big)\Big\|_Y
= \frac{\delta}{2 \| x \|_X} \Big\|   T(x) \Big\|_Y
\\
\Rightarrow & \quad \Big\|   Tx \Big\|_Y \le \frac{2}{\delta} \| x \|_X.
\\
\\
& \| Tv \|  < 1.
\\
\\
& h = \frac{\delta}{2} \frac{x}{\| x \|_X}, \quad \text{ with } \delta>0, x \in X, \text{and } x \ne 0.\\
& \rightarrow \Bigg\| \frac{\delta}{2} \frac{x}{\| x \|_X} \Bigg\| = \frac{\delta}{2}  \Bigg\|  \frac{x}{\| x \|_X} \Bigg\| = \frac{\delta}{2} < \delta\\
\\
& \| h \| = \Bigg\| \frac{\delta}{2} \frac{x}{\| x \|_X} \Bigg\| = \frac{\delta}{2}  \Bigg\|  \frac{x}{\| x \|_X} \Bigg\| = \frac{\delta}{2} < \delta.\\
 \| T(v+h) - Tv \| & = \| T(v+h-v) \|   \|\quad \text{(by linearity)}\\
 & = \| Th \| \\
& \le C  \| h \| \quad \text{(if $T$ is bounded).}\\
& \| Tx \|_{Y} \le C \| x \|_{X} \text{ for } \forall x \in X.\\
& \| h \| < 𝛿 \Rightarrow \| T(0+h) - T(0) \| < \epsilon. \\
\\
& \| T(h+x_0 - x_0) - T(0) \| < \epsilon \\
& \| T(h+x_0) - T(0 + x_0) \| < \epsilon \\
& \| T(x_0 + h) - T(x_0) \| < \epsilon. \\
\\
& \| h \| < 𝛿 \Rightarrow \| T(x_0 + h) - T(x_0) \| < \epsilon.\\

& A_g = \langle \cdot, g \rangle_X \text{ is a continuous linear functional.}\\
&\text{In a Hilbert space $\mathcal{F}$, for every continuous linear functional $L ∈ \mathcal{F}$ ',} \\
&\text{there exists a unique $g ∈ \mathcal{F}$, }\\
& L f ≡ \langle f, g \rangle_{\mathcal{F}}, \quad \quad \text{ for } \forall f \in \mathcal{F}. \\
& L: \mathcal{F} \rightarrow \mathbb{R}, \quad \quad L \in \mathcal{F}' \text{ (dual space)}. \\
&\text{If $\mathcal{F}$ is a normed space, the space $\mathcal{F}'$ of continuous linear functionals $L: \mathcal{F} → ℝ$ }\\
&\text{is called the topological dual space of $\mathcal{F}$.} \\
& \text{Let $M = Null(L) ⫋ \mathcal{F}$ be the null space for $L$,} \\
& \text{where }
Null(L) = \{f \in \mathcal{F} | Lf =0 \}.

\\
& \text{
$M$ is a closed linear subspace of $\mathcal{F}$, let's choose $h ∈ M^⊥$, with $\| h \|_\mathcal{F}$ = 1.
} \\
& u_f = (Lf)h - (Lh)f.\\
& Lu_f = L\Big( (Lf)h - (Lh)f \Big) = (Lf)Lh - (Lh)Lf = 0, \\
& \Rightarrow u_f \in M, \\
& \Rightarrow u_f ⊥ h, \text{ as } h \in M^⊥.\\
0 & = \langle u_f, h \rangle_{\mathcal{F}} \\
& = \Big\langle (Lf)h - (Lh)f, h \Big\rangle_{\mathcal{F}} \\
&= (Lf) \| h \|^2_{\mathcal{F}} - (Lh) \langle f, h  \rangle_{\mathcal{F}} \\
&= Lf - \Big\langle f, (Lh)h\Big\rangle_{\mathcal{F}}. \\
Lf & = \Big\langle f, (Lh)h\Big\rangle_{\mathcal{F}}. \\
& = \Big\langle f, g \Big\rangle_{\mathcal{F}}, \quad \text{where } g = (Lh)h. \\
\\
\end{align} \\
$$

$$
\begin{align} \\
& section 3 \\
&  k(x, y) = \langle \phi(x), \phi(y) \rangle_\mathcal{H}.\\
& \text{A function $k: X \times X \rightarrow \mathbb{R}$ is a kernel if there exists a Hilbert space $\mathcal{H}$ } \\
& \text{and a feaure map } 𝜙: X \rightarrow \mathcal{H}
\text{ such that} \\
\\
& \quad \quad \quad \quad \quad \quad k(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}, \quad \quad \forall x, x' \in X. \\
k(x, x') &= \exp\Big( - \gamma \| x - x' \|^2 \Big). \\
e^x & = 1 + \frac{1}{1!} x + \frac{1}{2!} x^2 +   \frac{1}{3!} x^3 + \cdots.\\
e^{xx'} & = 1 + \frac{1}{1!} xx' + \frac{1}{2!} x^2x'^2 +   \frac{1}{3!} x^3 x'^3  + \cdots. \\
& \exp\Big( - \gamma \| x - x' \|^2 \Big) \\
&= \exp\Big( - \gamma (x - x')^2 \Big) \\
&= \exp\Big( - \gamma x^2 + 2\gamma x x' - \gamma x'^2 \Big) \\
&= \exp\Big( - \gamma x^2 - \gamma x'^2 \Big)
\exp\Big(2\gamma x x' \Big)  \\
&= \exp\Big( - \gamma x^2 - \gamma x'^2 \Big) \Big( 1 + \frac{2 \gamma xx'}{1!}  + \frac{(2 \gamma xx')^2}{2!}  +   \frac{(2 \gamma xx')^3}{3!}  + \cdots \Big) \\
&= \exp\Big( - \gamma x^2 - \gamma x'^2 \Big) \Big( 1 \cdot 1  + \sqrt{\frac{2 \gamma}{1!}} x \cdot \sqrt{\frac{2 \gamma}{1!}} x'  + \\
& \quad \quad  \quad  \quad   \sqrt{\frac{(2 \gamma)^2}{2!}} x^2 \cdot \sqrt{\frac{(2 \gamma)^2}{2!}} x'^2 +
\sqrt{\frac{(2 \gamma)^3}{3!}} x^3 \cdot \sqrt{\frac{(2 \gamma)^3}{3!}} x'^3
+ \cdots \Big). \\
k(x, x') &= \exp\Big( - \gamma \| x - x' \|^2 \Big) = \Big\langle \phi(x),
\phi(x') \Big\rangle,\\
& \text{where } \phi(x) = \exp( - \gamma x^2) \Big(
1, \sqrt{\frac{2 \gamma}{1!}} x, \sqrt{\frac{(2 \gamma)^2}{2!}} x^2,
\sqrt{\frac{(2 \gamma)^3}{3!}} x^3, \cdots
  \Big)^\top. \\
& f(z) = \sum_{i=0}^\infty a_i z^i. \\
& k(x, x') = f \Big( \langle x, x' \rangle \Big) = \sum_{i=0}^\infty a_i  \langle x, x' \rangle^i.\\
&\forall n \ge 1, \forall a_1, a_2, \cdots, a_n \in \mathbb{R}^n,
\forall x_1, x_2, \cdots, x_n \in X, \\
& \text{Let $\mathcal{H}$ be a Hilbert space, $X$ a non-empty set and
$\phi: X \rightarrow \mathcal{H}$ a feature map}. \\
& \text{Then } k(x, x') = \langle \phi(x), \phi(x') \rangle \text{ is positive semidefinite.}\\

\\
\sum^n_{i=1}\sum^n_{j=1} a_i a_j k(x_i, x_j ) & =
 \sum^n_{i=1}\sum^n_{j=1} \Big\langle a_i \phi(x_i), a_j \phi(x_j)\Big\rangle_{\mathcal{H}}\\
& = \Bigg\| \sum^n_{i=1} a_i \phi(x_i)
\Bigg\|^2_{\mathcal{H}}
\\

&\ge 0. \\
& x^\top K x \ge 0, \quad \forall x \in \mathbb{R}^n. \\

 x^\top K x & = x^\top (Q^\top \Lambda Q) x \quad \quad \quad \text{(as $K$ is symmertrical).} \\
 &=( x^\top Q^\top) \Lambda (Qx) \\
 &= \sum_{i=1}^n \lambda_i (Qx)^\top_i (Qx)_i \\
 &= \sum_{i=1}^n \lambda_i y_i^2  \quad \quad \Big(\text{for } y_i = (Qx)_i \Big). \\

 & \ge 0  \quad \quad \text{ if $\forall \lambda_i \ge 0 $ }.  \\

\\
k(x, y) & =
\Bigg\langle
 \phi(x), \phi(y)
 \Bigg\rangle =
 \Bigg\langle
 \begin{pmatrix}
 1 \\
 \sqrt{2} x_1  \\
 \sqrt{2} x_2  \\
 \sqrt{2} x_1 x_2 \\
 x_1^2  \\
 x_2^2  \\
 \end{pmatrix},
 \begin{pmatrix}
 1 \\
 \sqrt{2} y_1  \\
 \sqrt{2} y_2  \\
 \sqrt{2} y_1 y_2 \\
 y_1^2  \\
 y_2^2  \\
 \end{pmatrix}
 \Bigg\rangle. \\
\phi(x) &=
\begin{pmatrix}
1 \\
\sqrt{2} x_1  \\
\sqrt{2} x_2  \\
\sqrt{2} x_1 x_2 \\
x_1^2  \\
x_2^2  \\
\end{pmatrix}. \\
\\
& f = f(\cdot) = [2, 1 , 3]^\top .
\\
& f(x) = 2x_1 + x_2 + 3 x_1 x_2, \\
\\
\end{align} \\
$$

$$
\begin{align} \\
& section 4 \\

& k(x, y)  =
\langle k(\cdot, x), k(\cdot, y) \rangle_\mathcal{H}  =
\langle \phi(x), \phi(y) \rangle_\mathcal{H}. \\
\\
& \phi(x) = k(\cdot, x) \in \mathcal{H}. \\
\\
&  \phi(x) = k(\cdot, x) \in \mathcal{H}, \quad \quad \quad x \in X.\\
&\langle k(\cdot, x), \phi(y) \rangle_\mathcal{H} =
\langle \phi(x), \phi(y) \rangle_\mathcal{H}, \\
&\langle \phi(x), k(\cdot, y)  \rangle_\mathcal{H} =
\langle \phi(x), \phi(y) \rangle_\mathcal{H}. \\

\\
f(x) & = f( \cdot)^\top \phi(x) = \Big\langle f( \cdot), \phi(x) \Big\rangle_{\mathcal{H}}.
\\
\\
f\Big( (-1, 4) \Big) & = \Bigg\langle
\begin{pmatrix}
2 \\
1 \\
3
\end{pmatrix}
,
\begin{pmatrix}
-1 \\
4 \\
-1 \times 4
\end{pmatrix}
 \Bigg\rangle_{\mathcal{H}}.

\\
\\
f(x) & =  e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots, \\
f(3) & =
\Bigg\langle
\begin{pmatrix}
1 \\
1 \\
\frac{1}{2!} \\
\frac{1}{3!} \\
\vdots
\end{pmatrix}
,
\begin{pmatrix}
1 \\
3 \\
3^2 \\
3^3 \\
\vdots
\end{pmatrix}
 \Bigg\rangle_{\mathcal{H}}.\\
 \\
& \quad \quad \quad f(\cdot) \quad  \phi(x) = k(\cdot, x) \\

& f(x) = \sum^n_{i=1} \alpha_i \Big\langle \phi(x_i), \phi(x) \Big\rangle, \quad \text{ where } \alpha_i \in \mathbb{R},  

\\
& f(x) = \sum^n_{i=1} \alpha_i \Big\langle  \phi(x), \phi(x_i) \Big\rangle, \quad \text{ where } \alpha_i \in \mathbb{R},  

\\
\end{align} \\

$$

$$
\begin{align} \\
& section 5 \\
f(x) & = \frac{1}{\sqrt{2 \pi} \sigma_f} \exp \Big( - \frac{(x - \mu_f)^2}{2 \sigma^2_f} \Big) \text{ and }
g(x) = \frac{1}{\sqrt{2 \pi} \sigma_g} \exp \Big( - \frac{(x - \mu_g)^2}{2 \sigma^2_g} \Big)
.\\

f(x)g(x) &= \frac{1}{2 \pi \sigma_f \sigma_g} \exp - \Big(  \frac{(x - \mu_f)^2}{2 \sigma^2_f} +  \frac{(x - \mu_g)^2}{2 \sigma^2_g} \Big).
\\
 f(x)g(x) & =  s \frac{1}{\sqrt{2 \pi} \sigma_{fg}} \exp \Big( - \frac{(x - \mu_{fg})^2}{2 \sigma^2_{fg}} \Big). \\
& p(y' | \boldsymbol x', \boldsymbol y, \boldsymbol X). \\
\| T_k f \|^2_2 & = \int^\infty_{-\infty} \mid T_k f(x) \mid^2 dx\\
&=  \int^{\infty}_{-\infty} \Bigg{|} \int^{\infty}_{-\infty} k(x, y) f(y) dy \Bigg{|}^2 dx
\\
& \le  \int^{\infty}_{-\infty} \Big(
\int^{\infty}_{-\infty}
\mid k(x, y) \mid^2 dy  \Big)
\Big(
\int^{\infty}_{-\infty}
\mid f(y) \mid^2 dy  \Big) dx \\
&  \quad  \quad  \text{(Cauchy-Schwarz inequality)}\\
& = \int^{\infty}_{-\infty}  \int^{\infty}_{-\infty}  \mid k(x, y) \mid^2 dy \text{  }
\| f \|^2_2 dx\\
& = \| f \|^2_2   \int^{\infty}_{-\infty}
 \int^{\infty}_{-\infty}  \mid k(x, y) \mid^2 dy dx\\
& = \| f \|^2_2 \| k \|^2_2 < \infty.  \\
Tf(x) & = \int^x_a k(x, y) f(y) dy, \quad \quad f \in C[a, b]. \\
Tf(x) & = \int^\infty_{-\infty} k(x, y) f(y) dy, \quad \quad f \in L^p.
\\
& \int^\infty_{-\infty} \int^\infty_{-\infty} \mid k(x, y) \mid^2 dx dy < \infty. \\
\langle f, f \rangle & = \sum^n_{i=1} \sum^n_{j=1} \alpha_i \alpha_j k(x_i, x_j) = \boldsymbol \alpha^\top \boldsymbol K \boldsymbol \alpha \ge 0. \\
& \langle f, g \rangle = \int_X f(x)g(x) dx.\\
& D = \{x_1, x_2, \cdots, x_n\}, \quad \quad x_i \in \mathbb{R}^m. \\
& k(\cdot, x) = k_x(\cdot), \quad \quad \quad \text{(a.k.a } f(\cdot)).\\
f(x) & = \sum^n_{i=1} \alpha_i k(x, x_i), \quad \quad \text{ as }
k(x, x_i) = \Big\langle
\phi(x), \phi(x_i)
\Big\rangle_{\mathcal{H}}, \\
 f(\cdot) & = \sum^n_{i=1} \alpha_i k(\cdot, x_i). \\
 \mathcal{F} & = \Bigg\{
\sum^n_{i=1} \alpha_i k(\cdot, x_i), \quad \alpha_i \in \mathbb{R}, x_i \in X
\Bigg\}. \\
& f, g  \in \mathcal{F} \text{ with } \\
& f(x) = \sum^l_{i=1} \alpha_i k(x, x_i) \quad \text{and} \quad
g(x) = \sum^n_{j=1} \beta_j k(x, x_j). \\
\\
(f + g)(x) & = f(x) + g(x), \\
\langle f, g \rangle_{\mathcal{H}} & = \sum^l_{i=1} \sum^n_{j=1} \alpha_i \beta_j k(x_i, x_j) =  \sum^l_{i=1} \alpha_i g(x_i) = \sum^n_{j=1} \beta_j f(x_j). \\
\langle f, f \rangle_{\mathcal{H}} & = \sum^l_{i=1} \sum^l_{j=1} \alpha_i \alpha_j k(x_i, x_j) = \boldsymbol \alpha_i^\top \boldsymbol K \boldsymbol \alpha_j \ge 0, \\
& \text{where } \alpha_i, \alpha_j \in \mathbb{R}, \boldsymbol K
\text{ is the kernel matrix constructed from } x_1, x_2, \cdots, x_l. \\
\Big\langle f, g \Big\rangle_{\mathcal{H}} & = \sum^l_{i=1} \alpha_i g(x_i).   \\
& \Big\langle f, k(x, \cdot) \Big\rangle_{\mathcal{H}} = \sum^l_{i=1} \alpha_i k(x, x_i)  = f(x).\\
&  k(x_i, x_j) = k\Big( \langle x_i, x_j\rangle  \Big) = f(x) = e^{x}, \quad \quad \text{ for } x = \langle x_i, x_j\rangle. \\
& \forall x \in X, \text{ there exists a } k(\cdot, x) \in \mathcal{H}
\\
& f(x) = \langle f, k( \cdot, x) \rangle_{\mathcal{H}},
\quad \quad \text{ for } \forall f \in \mathcal{H}.\\
K v & = \lambda v. \\
v^\top K v & = v^\top (\lambda v) = v^\top v \lambda = \lambda \| v \|^2, \\
v^\top K v & \ge 0 \text{ if } \lambda \ge 0. \\
& f(\cdot) = \phi(x_i) = k_{x_i}(\cdot) = k(\cdot, x_i). \\
& L_x: \mathcal{H} \rightarrow \mathbb{R}, \quad \text{where $\mathcal{H}$ is a Hilbert space of functions on $X$.}\\
& L_x(f) = f(x), \quad \quad \text{for } \forall x \in X, \quad \forall f \in \mathcal{H}. \\
& \phi(x) = k(・, x), \quad x \in X
\\
& f(x) = L_x(f) = \langle f, g \rangle_{\mathcal{F}} =
 \langle f, k_x \rangle_{\mathcal{F}}, \quad \text{for } x \in X, ∃ k_x \in {\mathcal{F}}, \forall f \in \mathcal{F}.\\
&  k_x \in {\mathcal{F}},\\
& f(y) = L_y(f) =
 \langle f, k_y \rangle_{\mathcal{F}} \rightarrow k_x(y) =
 L_y(k_x) =  \langle k_x, k_y \rangle_{\mathcal{F}}. \\
 k_x(y) & = \langle k_x, k_y \rangle_{\mathcal{F}}.\\
 k(x, y) & =  \langle k(\cdot, x), k(\cdot, y) \rangle_{\mathcal{F}}.
\\
& L_x(f) = f(x) =
 \langle f, k(\cdot, x) \rangle_{\mathcal{F}} \\
& f_z(\cdot) = \langle \cdot, z \rangle. \\
f(\cdot) & = \sum^n_{i=1} \alpha_i k(\cdot, x_i), \text{ or} \\
f(x) & = \sum^n_{i=1} \alpha_i k(x, x_i).\\
f(x) & = \Big\langle f, k(x, \cdot) \Big\rangle. \\
& k_{x_{i}} (\cdot) \text{ means the function $k$ is parameterized by } x_i. \\

\end{align}
$$

$$
\begin{align} \\
& section 6 \\

k(x, y) &= k_1(x, y) + k_2(x, y)\\
k(x, y) &= \alpha k_1(x, y), \quad \alpha \in \mathbb{R}\\
k(x, y) &= k_1(x, y) k_2(x, y)\\
k(x, y) &= f(x) f(y)\\
k(x, y) &= k_1(\phi(x), \phi(y))\\
k(x, y) &= x^\top B z, \quad \text{where $B$ is a symmetric positive semidefinite matrix.}  \\
\\
\text{Polynomials of degree exactly $d$: } \quad & k(x, y) = ⟨x, y ⟩^d, \\
 \text{Polynomials of degree up to $d$: } \quad & k(x, y) = \Big(⟨x, y ⟩ +1 \Big)^d. \\
\text{Gaussian kernels: } \quad &  k(x, y) =  \exp \Bigg(
  - \frac{‖x - y ‖^2_2 }{2 \sigma^2}
  \Bigg).\\
\text{Exponential kernels: } \quad & k(x, y) = \exp( \langle x, y \rangle).\\
\text{Exponentiated quadratic kernel: } \quad & k(x, y) = \exp \Big(

- \gamma^{-2} \| x - y \|^2  
\Big). \\
f(\cdot) & = 2 \cdot k(\cdot,  0) -3  \cdot k(\cdot,  1)
 +5 \cdot k(\cdot,  2) - 2 \cdot k(\cdot,  3) + 3
 \cdot k(\cdot,  4).\\

f(\cdot) & = \Bigg\langle
  \begin{pmatrix}
  2 \\
  -3 \\
  5 \\
  -2 \\
 3
  \end{pmatrix},
  \begin{pmatrix}
  k(\cdot, 0) \\
  k(\cdot, 1) \\
  k(\cdot, 2) \\
  k(\cdot, 3) \\
  k(\cdot, 4)
  \end{pmatrix}

\Bigg\rangle.

\\
& f(x) = \sum^{\infty}_{-\infty} f_l \exp(ilx). \\
\\
& f(x) = \sum^\infty_{l=0} 2 f_l \cos(lx), \quad \quad f_l = \frac{\sin(lT)}{l \pi}. \\
& f(x) =
2 \times \Bigg\langle
\begin{pmatrix}
0 \\
\frac{\sin(T)}{\pi} \\
\frac{\sin(2T)}{2\pi} \\
\frac{\sin(3T)}{3\pi} \\
\frac{\sin(4T)}{4\pi} \\
\vdots
\end{pmatrix}
\begin{pmatrix}
1 \\
\cos(x) \\
\cos(2x) \\
\cos(3x) \\
\cos(4x) \\
\vdots
\end{pmatrix}
\Bigg\rangle.\\

& k(x, y) = k(x-y) =  \exp \Big(

 - \gamma^{-2} \| x - y \|^2  
 \Big).\\
& \Big(T_kf \Big)(x) = \int_X k(x, x') f(x') dx'. \\
& k(x, x') = \sum^\infty_{l=1} \lambda_l e_l(x)  e_l(x'). \\
& k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}. \\
 k(x, y) & = \exp \Bigg(
- \frac{\| x- y \|^2_2}{\sigma^2}
\Bigg) \\
& = \exp \Bigg(
\frac{-\|x\|^2_2 -\|y\|^2_2 + 2 x^Ty }{\sigma^2}
\Bigg) \\
& =
\exp \Bigg(\frac{-\|x\|^2_2 }{\sigma^2} \Bigg)
\exp \Bigg(\frac{-\|y\|^2_2 }{\sigma^2} \Bigg)
\exp \Bigg(\frac{2x^Ty}{\sigma^2}\Bigg).
 \\
 \exp \Bigg(\frac{-\|x\|^2_2 }{\sigma^2} \Bigg)
 \exp \Bigg(\frac{-\|y\|^2_2 }{\sigma^2} \Bigg)
 & = f(x) f(y) = k(x, y ).\\
 \frac{2x^Ty}{\sigma^2} & = c f(x) f(y) = k_1(x, y).\\
k(x, y) & =  \exp\Big(k_1(x, y)\Big) \quad \text{ as } \\
\exp(x) & = 1 + x + \frac{x^2}{2!} + \cdots + \frac{x^i}{i!} + \cdots.\\
f(x) & = \langle f, k(\cdot, x)\rangle_{\mathcal{H}} = \sum^{\infty}_{l=1} f_l k_l(\cdot, x) =
\begin{bmatrix}
f_1 \\
f_2 \\
f_3 \\
\vdots
\end{bmatrix}
\begin{bmatrix}
\quad & \quad & \quad & \quad &  \\
\quad & \quad & \quad & \quad & \\
\quad & \quad & \quad & \quad & \\
\quad & \quad & \quad & \quad & \\
\quad & \quad & \quad & \quad & \\
\quad & \quad & \quad & \quad &  \\
\end{bmatrix}.\\
\\
\phi(x) & = k(\cdot, x). \\
\\
k_1(\cdot, x) & = \\ \\
k_2(\cdot, x) & = \\ \\
k_3(\cdot, x) & = \\ \\

f(x) & = \sum^{\infty}_{l=1} f_l k_l(\cdot, x) \\
& = \sum^{\infty}_{l=1} \Bigg(
  \sum^m_{i=1} \alpha_i k_l(\cdot, x_i)
  \Bigg) k_l(\cdot, x)\\
& = \Bigg\langle   \sum^m_{i=1} \alpha_i k_l(\cdot, x_i), k(\cdot, x)\Bigg\rangle_{\mathcal{H}} \\
&= \sum^m_{i=1} \alpha_i k_l(x, x_i).
\\
\end{align}
$$

$$
\begin{align}
& Section 7 \\
&\langle V, \mathbb{R}, +, \times, \boldsymbol 1,  \boldsymbol 0
 \rangle.\\
 \\
+ & :  \mathbb{V} \times \mathbb{V} \rightarrow \mathbb{V},\\
\times &:  \mathbb{R} \times  \mathbb{V} \rightarrow \mathbb{V}.\\
& 3x^1 + 4x^2, \\
& x+2x^2 + 0.5x^5, \\
& 5x^2. \\
x = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n \\
\end{bmatrix},  
y = \begin{bmatrix}
y_1 \\
y_2 \\
y_3 \\
\vdots \\
y_n \\
\end{bmatrix}, &
x +y =
\begin{bmatrix}
x_1 + y_1 \\
x_2 + y_2 \\
x_3 + y_3 \\
\vdots \\
x_n + y_n \\
\end{bmatrix},
\lambda x =
\begin{bmatrix}
\lambda x_1 \\
\lambda x_2 \\
\lambda x_3 \\
\vdots \\
\lambda x_n \\
\end{bmatrix}.
\\
y & = \theta x. \\
\boldsymbol y & = \boldsymbol X \boldsymbol \theta . \\
\\
J(\boldsymbol \theta)  & = \frac{1}{2} \| \boldsymbol X \boldsymbol \theta - \boldsymbol  y \|^2\\

 & = \frac{1}{2} (\boldsymbol X \boldsymbol \theta - \boldsymbol  y)^\top (\boldsymbol X \boldsymbol \theta - \boldsymbol  y)
 \\
& =  \frac{1}{2} (\boldsymbol \theta^\top X^\top - \boldsymbol y^\top) (\boldsymbol X \boldsymbol \theta - \boldsymbol  y) \\
&= \frac{1}{2} \Big( \boldsymbol \theta^\top X^\top \boldsymbol X \boldsymbol \theta
- (\boldsymbol X \boldsymbol \theta)^\top \boldsymbol y
- \boldsymbol y^\top (\boldsymbol X \boldsymbol \theta) + \boldsymbol y^\top \boldsymbol y \Big)\\
&=  \frac{1}{2} \boldsymbol \theta^\top X^\top \boldsymbol X \boldsymbol \theta
- (\boldsymbol X \boldsymbol \theta)^\top \boldsymbol y +
\frac{1}{2} \boldsymbol y^\top \boldsymbol y\\
& \quad \text{ since } (\boldsymbol X \boldsymbol \theta)^\top \boldsymbol y =  \text{scalar} = \text{scalar}^\top =  \Big((\boldsymbol X \boldsymbol \theta)^\top \boldsymbol y\Big)^\top =  \boldsymbol y^\top (\boldsymbol X \boldsymbol \theta). \\
& \frac{d J(\boldsymbol \theta)}{d \boldsymbol \theta}
= \boldsymbol X^\top X \boldsymbol \theta -  \boldsymbol X^\top \boldsymbol y = 0 \\
\Rightarrow & \boldsymbol X^\top \boldsymbol X \boldsymbol \theta =  \boldsymbol X^\top \boldsymbol y \\
\Rightarrow & \boldsymbol \theta  = \Big(\boldsymbol X^\top \boldsymbol X \Big)^{-1}  \boldsymbol X^\top \boldsymbol y. \\
& \boldsymbol \theta  = \Big(\boldsymbol X^\top \boldsymbol X \Big)^{-1}  \boldsymbol X^\top \boldsymbol y. \\
\boldsymbol A \boldsymbol x&  = \boldsymbol b, \quad \text{ where }
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{bmatrix}. \\
\text{Length/norm: } & \quad \| x \|^2  = \langle x, x \rangle,\\
\text{Distance: } & \quad d(x, y) = \| x - y \|,\\
\text{Angle: } & \quad \cos(\theta) = \frac{\langle x, x \rangle}{\|x\| \|y\|}. \\
| \langle x, y \rangle | & \le \| x \| \| y \|, \\
\|x + y\| & \le \| x \| + \| y \|.\\
y^{(i)} & = \theta^\top x^{(i)} + \epsilon^{(i)}, \quad \quad  \quad i = 1, \cdots, n. \\ \\
\theta \sim & \mathcal{N} (0, \tau^2 I ),  \quad \quad  \quad \theta \in \mathcal{R}^m.\\
\epsilon\sim & \mathcal{N} (0, \sigma^2 ),  \quad \quad \text{ } \text{ }  \quad \theta \in \mathcal{R}^n.
\\ \\
\frac{\partial \boldsymbol y}{\partial \boldsymbol x} & =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & & \vdots\\
\frac{\partial  y_m}{\partial  x_1} & \cdots & \frac{\partial  y_m}{\partial x_n}  \\
\end{bmatrix}, \quad \frac{\partial \boldsymbol y}{\partial \boldsymbol x}  = \boldsymbol A, \quad \text{for } \boldsymbol{Ax} = \boldsymbol y. \\ \\


H & =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial  x_1 \partial  x_n}  \\
\frac{\partial^2 f}{\partial x_2 x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial  x_2 \partial  x_n}  \\
\vdots & & & \vdots\\
\frac{\partial^2 f}{\partial  x_n \partial x_1} & \frac{\partial^2 f}{\partial  x_n \partial  x_2} & \cdots & \frac{\partial^2 f}{\partial  x_n^2}  \\
\end{bmatrix}. \\ \\





\end{align}
$$

$$
\begin{align}
& Section 8 \\ \\
& \frac{\partial \boldsymbol x^\top  \boldsymbol a}{\partial \boldsymbol x} = \frac{\partial \boldsymbol a^\top  \boldsymbol x}{\partial \boldsymbol x} = \boldsymbol a^\top, \\ \\
& \frac{\partial \boldsymbol A  \boldsymbol x}{\partial \boldsymbol x} = \boldsymbol A, \quad  \frac{\partial   \boldsymbol x^\top \boldsymbol A}{\partial \boldsymbol x} = \boldsymbol A^\top, \\ \\
& \frac{\partial \boldsymbol A  \boldsymbol u}{\partial \boldsymbol x}
= \boldsymbol A \frac{\partial  \boldsymbol u}{\partial \boldsymbol x}, \\ \\
& \frac{\partial \boldsymbol x^\top  \boldsymbol x}{\partial \boldsymbol x} = 2 \boldsymbol x^\top, \\ \\
& \frac{\partial \boldsymbol b^\top \boldsymbol A  \boldsymbol x}{\partial \boldsymbol x} = \boldsymbol b^\top \boldsymbol A, \\ \\
& \frac{\partial \boldsymbol x^\top \boldsymbol A  \boldsymbol x}{\partial \boldsymbol x} = \boldsymbol x^\top \Big(\boldsymbol A + \boldsymbol A^\top \Big), \\ \\
& \frac{\partial \boldsymbol x^\top \boldsymbol A  \boldsymbol x}{\partial \boldsymbol x} = 2 \boldsymbol x^\top \boldsymbol A, \quad \text{if } \boldsymbol A \text{ is symmetric,} \\ \\
& \frac{\partial^2 \boldsymbol x^\top \boldsymbol A  \boldsymbol x}{\partial \boldsymbol x \partial \boldsymbol x^\top} = \boldsymbol A + \boldsymbol A^\top, \\ \\
& \frac{\partial \boldsymbol a^\top \boldsymbol{xx}^\top  \boldsymbol b}{\partial \boldsymbol x} = \boldsymbol x^\top \Big( \boldsymbol {ab}^\top + \boldsymbol{ba}^\top\Big). \\ \\
\\
\\
& \frac{\partial f(\boldsymbol X)^\top}{\partial \boldsymbol X}
= \Bigg ( \frac{\partial f(\boldsymbol X)}{\partial \boldsymbol X} \Bigg)^\top
\\
& \frac{\partial \boldsymbol a^\top \boldsymbol{X} \boldsymbol b}{\partial \boldsymbol X} = \boldsymbol{ba}^\top, \\ \\
& \frac{\partial \boldsymbol a^\top \boldsymbol{X}^\top \boldsymbol b}{\partial \boldsymbol X} = \boldsymbol{ab}^\top, \\ \\
& \frac{\partial \boldsymbol a^\top \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol b}{\partial \boldsymbol X} = \Big(  \boldsymbol{ab}^\top  + \boldsymbol{ba}^\top \Big)\boldsymbol{X}^\top, \\ \\
& \frac{\partial \boldsymbol{X}^\top \boldsymbol A \boldsymbol{X}}{\partial \boldsymbol X} = \boldsymbol{X}^\top \Big(
  \boldsymbol A + \boldsymbol A^\top
  \Big). \\ \\
\\  
  & \frac{\partial \boldsymbol u^\top \boldsymbol v}{\partial \boldsymbol x} =   \boldsymbol u^\top
  \frac{\partial \boldsymbol v}{\partial \boldsymbol x} +
  \boldsymbol v^\top \frac{\partial \boldsymbol u}{\partial \boldsymbol x},
\\ \\
& \frac{\partial g(\boldsymbol u)}{\partial \boldsymbol x} =   
\frac{\partial g(\boldsymbol u)}{\partial \boldsymbol u}
\frac{\partial \boldsymbol u}{\partial \boldsymbol x}. \\
\langle  a, b \rangle & = a^\top  b. \\
Ax & = b \\
\boldsymbol A^\top \boldsymbol{Ax} & = \boldsymbol  A^\top \boldsymbol  b \\
\boldsymbol x & = (\boldsymbol  A^\top \boldsymbol  A)^{-1} \boldsymbol A^\top \boldsymbol  b. \\
\boldsymbol x & = \sum_{i=1}^k \alpha_i \boldsymbol b_i.\\ \\
T(\lambda \boldsymbol x) & = \lambda T(\boldsymbol x), \\
T(\boldsymbol x + \boldsymbol y) & = T(\boldsymbol x) + T(\boldsymbol y). \\

1 \cdot \boldsymbol u' & = a \cdot \boldsymbol u + b \cdot \boldsymbol w. \\
\\
\boldsymbol B & = \Bigg\{
  \begin{bmatrix}
  1 \\
  0 \\
  \end{bmatrix},
  \begin{bmatrix}
  0 \\
  1 \\
  \end{bmatrix}
\Bigg\} \text{ and }
\boldsymbol B' = \Bigg\{
  \begin{bmatrix}
  4 \\
  1 \\
  \end{bmatrix},
  \begin{bmatrix}
  -3 \\
  2 \\
  \end{bmatrix}
\Bigg\}.
\\
\boldsymbol P & =   \begin{bmatrix}
  4 & -3 \\
  1 & 2 \\
  \end{bmatrix}.
\\
\\
[\boldsymbol  v]_{\boldsymbol B'} & = \begin{bmatrix}
  1 \\
 2 \\
  \end{bmatrix}.
\\
 \\
 [\boldsymbol v]_{\boldsymbol B} & =
 \begin{bmatrix}
   4 & -3 \\
   1 & 2 \\
   \end{bmatrix}
 \begin{bmatrix}
   1 \\
  2 \\
   \end{bmatrix}_{\boldsymbol B'}
=
\begin{bmatrix}
  -2 \\
 5 \\
  \end{bmatrix}_{\boldsymbol B} .

 \\ \\
 \begin{bmatrix}
   1 \\
  2 \\
   \end{bmatrix}_{\boldsymbol B'}
& =
\begin{bmatrix}
  -2 \\
 5 \\
  \end{bmatrix}_{\boldsymbol B}

\\
\boldsymbol u' & = a \boldsymbol u + b \boldsymbol w, \\
\boldsymbol w' & = c \boldsymbol u + d \boldsymbol w. \\

[\boldsymbol v]_B = \boldsymbol P[\boldsymbol v]_{\boldsymbol B'} & =
\begin{bmatrix}
a & c \\
b & d \\
\end{bmatrix} [\boldsymbol v]_{\boldsymbol B'}. \\
& A  = [a_1, a_2, \cdots, a_n] \text{ where $a_i$ is the $i$th column of $A$}. \\
& \text{Im}(T)  = \text{span}[a_1, a_2, \cdots, a_n]. \\
\\
\\
\boldsymbol{Ax} & = \boldsymbol a_1 x_1 + \boldsymbol a_2 x_2 + \cdots + \boldsymbol a_n x_n \\
\boldsymbol{Ax} & = \begin{bmatrix}
A_{11} \\
A_{21} \\
\vdots \\
A_{n1} \\
\end{bmatrix} x_1 +
\begin{bmatrix}
A_{12} \\
A_{22} \\
\vdots \\
A_{n2} \\
\end{bmatrix} x_2 + \cdots +
\begin{bmatrix}
A_{1n} \\
A_{2n} \\
\vdots \\
A_{nn} \\
\end{bmatrix} x_n .
\\
\\
\\
\begin{bmatrix}
a & c \\
b & d \\
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
\end{bmatrix}_{\boldsymbol B'} & =
\begin{bmatrix}
a \\
b \\
\end{bmatrix}_{\boldsymbol B}.
\\
\\
\begin{bmatrix}
  1 \\
 0 \\
  \end{bmatrix}_{\boldsymbol B'}
=
\begin{bmatrix}
  4 \\
 1 \\
  \end{bmatrix}_{\boldsymbol B}
  \\
  \\
T(\boldsymbol x) = \boldsymbol {Ax}.  
\\ \\
A' & = \begin{pmatrix}
1 & 0 \\
0 & -1 \\
\end{pmatrix}, \quad A =
\begin{pmatrix}
2 & -3 \\
1 & -2 \\
\end{pmatrix}, \quad P =
\begin{pmatrix}
3 & 1 \\
1 & 1 \\
\end{pmatrix},\\
A' &= P^{-1}A P,
\\
\\
\begin{pmatrix}
1 & 0 \\
0 & -1 \\
\end{pmatrix}
& =
\begin{pmatrix}
3 & 1 \\
1 & 1 \\
\end{pmatrix}^{-1}
\begin{pmatrix}
2 & -3 \\
1 & -2 \\
\end{pmatrix}
\begin{pmatrix}
3 & 1 \\
1 & 1 \\
\end{pmatrix}.\\
\begin{pmatrix}
3 & 1 \\
1 & 1 \\
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & -1 \\
\end{pmatrix}
& =
\begin{pmatrix}
2 & -3 \\
1 & -2 \\
\end{pmatrix}
\begin{pmatrix}
3 & 1 \\
1 & 1 \\
\end{pmatrix}.\\
\end{align}
$$

$$
section 9 \\ \\

\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9 \\
\end{pmatrix}
\begin{pmatrix}
x_1\\
x_2\\
x_3\\
\end{pmatrix} =
\begin{pmatrix}
1  \\
4  \\
7  \\
\end{pmatrix} x_1 +
\begin{pmatrix}
2  \\
5  \\
8  \\
\end{pmatrix} x_2 +
\begin{pmatrix}
2
\begin{pmatrix}
2  \\
5  \\
8  \\
\end{pmatrix} -
\begin{pmatrix}
1  \\
4  \\
7  \\
\end{pmatrix}
\end{pmatrix} x_3 =
\begin{pmatrix}
1  \\
4  \\
7  \\
\end{pmatrix} x'_1 +
\begin{pmatrix}
2  \\
5  \\
8  \\
\end{pmatrix} x'_2.
\\  \\

\begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{pmatrix}
\begin{pmatrix}
x_1\\
x_2\\
x_3\\
\end{pmatrix} =
\begin{pmatrix}
a_{11} \\
a_{21} \\
a_{31}  \\
\end{pmatrix} x_1 +
\begin{pmatrix}
a_{12} \\
a_{22} \\
a_{32}  \\
\end{pmatrix} x_2 +
\begin{pmatrix}
a_{13} \\
a_{23} \\
a_{33}  \\
\end{pmatrix} x_3. \\ \\
(A + B)^{-1} \ne A^{-1} + B^{-1}. \\
T(x) = T(y) \rightarrow x = y, \quad x, y \in F, T(x), T(y) \in G. \\
\\
T(F) = G.\\
\\
\\
(A')^n  = P^{-1}AP (P^{-1}AP) \cdots P^{-1}AP = P^{-1}A^nP. \\
\\

$$
