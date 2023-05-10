$$
\begin{align}
 p(x | \theta) & = h(x) \exp( \theta^T T(x) - A (\theta)),
\\

\\
p(\theta | \gamma) & = h_c(\theta) \exp \Bigg(
  \begin{pmatrix}
  \gamma_1  &
  \gamma_2
  \end{pmatrix}

  \begin{pmatrix}
  \theta  \\
  - A(\theta)
  \end{pmatrix}
  - A_c(\gamma)
\Bigg). \\
p(x|\theta) & = \exp(x \log \frac{\theta}{1- \theta} + \log(1 - \theta)).
\\
\gamma & = \begin{pmatrix}
\alpha \\
\alpha + \beta
\end{pmatrix} \text{ and } h_c(\theta) = \frac{\theta}{1- \theta}.
\\
\\
p(\theta | \alpha, \beta) &= \frac{\theta}{1- \theta} \exp \Big( \alpha \log \frac{\theta}{1- \theta} + (\alpha + \beta) \log(1-\theta) - A_c(\gamma) \Big) \\
& = \exp \Big( (\alpha-1) \log  \theta + (\beta - 1) \log(1-\theta) - A_c(\alpha, \beta) \Big) \\
& ∝ \theta^{\alpha-1} (1-\theta)^{\beta - 1}
.
\\
\\
\theta & = \frac{v}{N}, \quad v \text{ occurances out of } N \text{ trials.}
  \\
  p(x | N, \theta) &
  =
  \begin{pmatrix}
  N  \\
  x
  \end{pmatrix}
  \theta^x (1-\theta)^{N-x}. \\
  p(x | N, \theta) & \rightarrow
\frac{e^{-\lambda t } (\lambda t)^x}{x!} \text{ when } v ≪ N \\
& \quad \quad \quad \quad  \quad \quad \text{ where } v = \lambda t, \lambda \text{ is the event rate, and }
\\
& \quad \quad \quad \quad  \quad \quad
  \text{ } t \text{ is the time interval}.
\\
  \mathbb{E} [X] & = \frac{1}{\theta}.
\\  
\mathbb{E} [X] & = \int x f(x) dx =  \int_x x  \frac{e^{-\lambda t }(\lambda t)^x}{x!},
\\  
\\
  \mathbb{E} [X_i] & = \frac{\alpha_i}{\sum_j \alpha_j} = \tilde{\alpha_i}, \\
  Var(X_i) & = \frac{\tilde{\alpha_i} (1 - \tilde{\alpha_i})}{\sum_j \alpha_j + 1}. \\
  \\


  \\
    \mathbb{E} [x] & = \frac{\alpha}{\beta},
    \\
    Var(x) & = \frac{\alpha}{\beta^2}. \\
    \\

\\
  \theta, 1-\theta \quad  \quad& \rightarrow \quad  \theta_1, \theta_2, \cdots, \theta_K  
\\

p(x, y | z) & = p(x| y, z) p(y |z). \\
\mathbb{E} [\theta] & = \frac{\alpha}{\alpha+\beta},
\\
Var(\theta) &= \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.
\\

p(\theta | x, N, \alpha,\beta) & ∝ p( x, |  N, \theta) p(\theta|  \alpha,\beta) \\
& ∝ \theta^x (1- \theta)^{N-x} \theta^{\alpha -1} (1- \theta)^{\beta-1}\\
& ∝ \theta^{x+\alpha-1} (1- \theta)^{N-x +\beta -1}  \\
p(\theta | x, N, \alpha,\beta) &= Beta( x + \alpha, N-x + \beta). \\

\\
p(\theta | \alpha,\beta) & = \frac{\Gamma(\alpha +
\beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha - 1}(1 - \theta)^{\beta - 1},
\\
Γ(z) & = \int^∞_0 x^{z-1} e^{-x} dx, \quad \quad z>0, \\
B(\alpha,\beta) & =\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha +
\beta)} \text{ where } B \text{ is the Beta function}.
\\
\\
& X  \sim Cat(\theta_1, \theta_2, \theta_3, \cdots , \theta_k), \\
& P(x_i) = \theta_i. \\
\\
P(X=1) & = \theta, \\
P(X=0) & = 1- \theta, \\
p(x | \theta) & = \theta^x (1-\theta)^{1-x}. \\
\\
& \theta, 1 - \theta \quad \rightarrow \quad \theta_1, \theta_2, \cdots \theta_k.
\\

p(x_1, x_2, \cdots, x_k | N, \theta_1, \theta_2, \cdots, \theta_k) &= \frac{N!}{x_1! \cdots x_k!} \theta_1^{x_1} \theta_2^{x_2} \cdots \theta_k^{x_k} .
\\
\mathbb{E} [x] & = \theta, \\
Var(x) & = \theta (1 - \theta). \\
\\

p(x | N, \theta) &
=
\begin{pmatrix}
N  \\
x
\end{pmatrix}
\theta^x (1-\theta)^{N-x}. \\
\\
\mathbb{E} [x] & = N \theta, \\
Var(x) & = N \theta (1 - \theta). \\
\\
\mathbb{E} [X_i] & = N \theta_i, \\
Var(X_i) & = N \theta_i (1 - \theta_i). \\
\\


p_X(x) & = \sum_y p(x, y) = \sum_y p(x | y) p(y) = \mathbb{E}_Y  [p_{X,Y}(x | y)].  \\

p(x|y) & = p(x), \\
p(x, y) & = p(x) p(y), \\
Cov(x, y) &= 0. \\
& \text{Define: } ⟨X,Y⟩  = Cov(x, y), \\
& ‖ X ‖ =  \sqrt{Cov(x, x)}.  \\
& \cos \theta = \frac{⟨x,y⟩}{‖ x‖ ‖ y‖} = \frac{Cov(x, y)}{ \sqrt{Var(x) Var(y)}}.
\\
\end{align}
$$


$$
\begin{align}
& Section 2 \\
f(x) &= f(x_0) + \frac{f'(x_0)}{1!}(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + ... \\
F_X(x) &= P(X_1≤x_1, X_2≤x_2, ... ) \quad \text{ for discret variable.}\\
F_X(x) &= \int_{-∞}^{x_1} \int_{-∞}^{x_2} f(z_1, z_2, ... ) d z_1 dz_2 \dotsb \quad \text{ for continuous variable.}
\\
\end{align}
$$

$$
\begin{align}
& Section 3 \\
& \text{A random variable is capital } X \text{ with values in small } x. \\
& \boldsymbol{A, B} \text{ in bold is a matrix.}\\
& \boldsymbol{x, y} \text{ in bold is a vector.} \\
\\
& P(X = x) = p(x). \\
& P_{X, Y}(X=x,Y=y)=P_{X,Y}(x,y).

\\
\\
&  \mathbb{E} [f(X)] = \mathbb{E}_{X}[f(X)] = \mathbb{E}_{X∼P(X)}[f(X)].  \\
& \mathbb{E} [f(X, Y)] = \mathbb{E}_{X, Y}[f(X, Y)] = \mathbb{E}_{X∼P(X), Y∼P(Y)}[f(X, Y)].\\
\\
&  \mathbb{E}_X [f(x)] = \int_X p(x) f(x)  dx.\\
&  \mathbb{E}_X [f(x)] = \sum_{x ∈ X} p(x) f(x). \\
\end{align}
$$

$$
\begin{align}
Section 4 \\
\rho_{X, Y} & = Corr(X, Y) = \frac{Cov(X, Y)}{\sqrt{Var(X)} \sqrt{Var(Y)}} =  \frac{ \mathbb{E} [(X - \mu_X)(Y - \mu_Y)] }{\sqrt{Var(X)} \sqrt{Var(Y)}$}.
\\
Σ_{ij} & = Cov(x_i, x_j)
\\
Var(x) & = \mathbb{E} \Big[(x - \mathbb{E}[x])^2 \Big]. \\
Cov(x, y) & = \mathbb{E} \Big[(x - \mathbb{E}[x])(y - \mathbb{E}[y]) \Big]. \\
Cov(\boldsymbol{x}, \boldsymbol{y}) & = \mathbb{E} \Big[(\boldsymbol{x} - \mathbb{E}[\boldsymbol{x}])(\boldsymbol{y} - \mathbb{E}[\boldsymbol{y}])^T \Big] ∈ R^{m \times n} \text{ for vectors } \boldsymbol{x}  ∈ R^{m}, \boldsymbol{y}  ∈ R^{n}.\\
\\

 Var(x) &= \int x^2 f(x) dx - \mu^2 \quad \text{ where } f(x) \text{ is the probability density function, and } \\
& \quad \quad \quad\quad \quad \quad \quad \quad \quad \quad μ \text{
   is the expected value for } x. \\
\\
p(x, y) & = p(x|y) p(y). \\
 p(x_1, x_2, \cdots, x_k ) &= p(x_1) p(x_2|x_1) p(x_3 | x_2, x_1) \cdots p(x_k | x_{k-1} \cdots x_1). \\
\\
p(e) & = \int_{h} p(e|h) p(h) dh,  \text{ or } \\
&
= \mathbb{E}_{h∼P(H)} [p(e|h)]. \\
\\
\end{align}
$$

$$
\begin{align}
& Section 5 \\
\mathbb{E}[X] & =\frac{\alpha}{\beta}, \\  
Var(X) & = \frac{\alpha}{\beta^2}. \\  
\mathbb{E}[X] & =\frac{1}{\lambda}, \\  
Var(X) & = \frac{1}{\lambda^2}. \\  
\mathbb{E}[X] & = Var(X) = \lambda t. \\  
\mathbb{E}[X] & = Var(X) = \lambda t. \\  
\mathbb{E} [a f(x) + b g(x)] & = a \mathbb{E} [f(x)] + b\mathbb{E} [g(x)]. \\
\\
Var(x) &= \mathbb{E}[x^2] - \mathbb{E}[x]^2. \\
Var(\boldsymbol{x}) &= \mathbb{E}[\boldsymbol{xx}^T] - \mathbb{E}[\boldsymbol{x}]\mathbb{E}[\boldsymbol{x}]^T. \\
Var(a) &= 0. \\
Var(ax + b) &= a^2 Var(x). \\
Var(x + y) &= Var(x) + Var(y) + 2 Cov(x, y). \\
Var(\boldsymbol{A} \boldsymbol{x + b}) &= \boldsymbol{A} Var(\boldsymbol{x}) \boldsymbol{A}^T = \boldsymbol{A} \boldsymbol{Σ} \boldsymbol{A}^T. \\
Var(\boldsymbol{x + y}) &= Var(\boldsymbol{x}) + Var(\boldsymbol{y}) + Cov(\boldsymbol{x}, \boldsymbol{y}) + Cov(\boldsymbol{y}, \boldsymbol{x}).\\
\\
Cov(x, y) &= \mathbb{E}[xy] - \mathbb{E}[x]\mathbb{E}[y]. \\
Cov(x, y)  &= Cov(y, x).   \\
Cov(x, x)  &= Var(x).   \\
\\
P(A | B) & = \frac{P(B|A) P(A)}{P(B)}. \\
\\
P(a ≤ X ≤ b) &= \int_a^b f(x) dx. \\
P(X ≤ a) &= \int_{-∞}^b f(x) dx. \\
\\
P \Big( P \left( X  \text{=head}\right) = θ  \Big), & \quad \text{ where } θ ∈ [0, 1]. \\

\\
f(x) & = \sum_{k=0}^\infty c_k x^k = c_0 + c_1 x + c_2 x^2 + \dotsb \\.
&\text{With ML, we predict a score } R =r(x)\\

\end{align}
$$

$$
\begin{align}
& Section 6 \\
&  \mathcal{N}\Big(μ, \frac{σ²}{n} \Big). \\
p(\boldsymbol{x}| \boldsymbol{\mu}, \boldsymbol{\Sigma}) \\
p(\boldsymbol{x})& = \mathcal{N}(\boldsymbol{x}| \boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad X ~ \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}).
\\
\text{For } Y & \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), X \sim \mathcal{N}(0, I)
\text{, then} \\
\boldsymbol y & = \boldsymbol {A x + \mu}, \text{ where }  \boldsymbol{A A}^T =  \boldsymbol \Sigma.
\\
p(\boldsymbol{x}) & =  \mathcal{N}(\boldsymbol{\mu_\boldsymbol{x}}, \boldsymbol{\Sigma_\boldsymbol{xx}}), \quad p(\boldsymbol{y}) =  \mathcal{N}(\boldsymbol{\mu_y}, \boldsymbol{\Sigma_\boldsymbol{yy}}).
\\
\\
p(\boldsymbol{x}, \boldsymbol{y}) & =
\mathcal{N} \Big(\begin{bmatrix}
\boldsymbol{\mu_x}  \\
\boldsymbol{\mu_y}
\end{bmatrix} ,
\begin{bmatrix}
\boldsymbol{\Sigma_\boldsymbol{xx}} & \boldsymbol{\Sigma_\boldsymbol{xy}}  \\
\boldsymbol{\Sigma_\boldsymbol{yx}} & \boldsymbol{\Sigma_\boldsymbol{yy}}
\end{bmatrix}
  \Big).
\\
\\
\boldsymbol{\Sigma} & =
\begin{pmatrix}
Cov(x_1, x_1) & Cov(x_1, x_2) & \cdots & Cov(x_1, x_n) \\
Cov(x_2, x_1) & Cov(x_2, x_2) & \cdots & Cov(x_2, x_n) \\
\vdots  & \vdots  & \ddots & \vdots  \\
Cov(x_n, x_1) & Cov(x_n, x_2) & \cdots & Cov(x_n, x_n) \\

\end{pmatrix}.
\\
\\
p(\boldsymbol{x} | \boldsymbol{y}) & = \mathcal{N}(\boldsymbol{\mu_{x|y}}, \boldsymbol{\Sigma_{x|y}}). \\
\\
\boldsymbol{\mu_{x|y}} &=\boldsymbol{\mu_x}+\boldsymbol{\Sigma_{xy}}{\boldsymbol{\Sigma_{yy}}}^{-1}({\boldsymbol y}-\boldsymbol{\mu_y}),
\\
\boldsymbol{\Sigma_{x|y}} &=\boldsymbol{\Sigma_{xx}}-\boldsymbol{\Sigma_{xy}}{\boldsymbol{\Sigma_{yy}}}^{-1}\boldsymbol{\Sigma_{yx}}
\quad  \text{ where }  
  \boldsymbol{\Sigma_{xy}} = Cov(\boldsymbol x, \boldsymbol y).
\\
\\
\end{align}
$$


$$
\begin{align}
\\& Section 7 \\

p(x_1, x_2) & =
\mathcal{N} \Big(\begin{bmatrix}
0  \\
3  
\end{bmatrix} ,
\begin{bmatrix}
7 & 2  \\
2 & 1
\end{bmatrix}
  \Big).

\\
\mu_{x_1 | x_2=-2} & =  \mu_x+{\sigma^2}_{x_1x_2}{{\sigma^2}_{x_2x_2}}^{-1}({ x_2}-\mu_{x_2}) = 0 + 2 \cdot 1^{-1} (-2 - 3) = -10, \\
\sigma^2_{x_1 | x_2=-2} & = \sigma^2_{x_1x_1}-\sigma^2_{x_1x_2}{\sigma^2_{x_2x_2}}^{-1}\sigma^2_{x_2x_1}= 7 - 2 \cdot 1^{-1} \cdot 2 = 3. \\
\\
p(x_1 | x_2 = -2) & =
\mathcal{N} (-10, 3). \\
p(\boldsymbol x) & = \int p(\boldsymbol x, \boldsymbol y) d \boldsymbol y = \mathcal{N} (\boldsymbol x | \boldsymbol \mu_{\boldsymbol x}, \boldsymbol \Sigma_{\boldsymbol{xx}} ).
\\
\mathcal{N} (\boldsymbol x | \boldsymbol a, \boldsymbol A)  \mathcal{N} (\boldsymbol x | \boldsymbol b, \boldsymbol  B) & = s \mathcal{N} (\boldsymbol x | \boldsymbol c, \boldsymbol C) \text{ where } s ∈ \mathbb{R}, \boldsymbol x ∈ \mathbb R^D.  \\
\boldsymbol C & = (\boldsymbol A^{-1} + \boldsymbol B^{-1})^{-1}, \\
\boldsymbol c & = \boldsymbol C (\boldsymbol A^{-1} \boldsymbol a + \boldsymbol B^{-1} \boldsymbol b),  \\
s &= (2 \pi)^{-\frac{D}{2}} | \boldsymbol A + \boldsymbol B|^{-\frac{1}{2}} \exp(-\frac{1}{2} (\boldsymbol a - \boldsymbol b)^T (\boldsymbol A + \boldsymbol B)^{-1}(\boldsymbol a - \boldsymbol b)). \\
\\
P(H|E) & = \frac{P(E|H) P(H)}{P(E)} = \frac{1}{P(E)} \mathcal{N} (\boldsymbol e_{|h} | \boldsymbol a, \boldsymbol A) \mathcal{N} (\boldsymbol h | \boldsymbol b, \boldsymbol B) = \mathcal{N} (\boldsymbol h_{|e} | \boldsymbol c, \boldsymbol C) \\
\\
& \text{ as } P(H|E) \text{ is a PDF.}
\\
p(\boldsymbol x, \boldsymbol y) &= \mathcal{N} (\boldsymbol \mu_{\boldsymbol x} + \boldsymbol \mu_{\boldsymbol y}, \boldsymbol \Sigma_{\boldsymbol x} + \boldsymbol \Sigma_{\boldsymbol y}).
\\
p(a \boldsymbol x, b \boldsymbol y) &= \mathcal{N} (a \boldsymbol \mu_{\boldsymbol x} + b \boldsymbol  \mu_{\boldsymbol y}, a^2\boldsymbol \Sigma_{\boldsymbol x} + b^2 \boldsymbol \Sigma_{\boldsymbol y}).
\\
\\
& \text{With } \boldsymbol y = \boldsymbol {A x}, \text{ and }  X ∼ \mathcal{N} (\boldsymbol \mu, \boldsymbol \Sigma), \\
&\text { we can calculate } \mathbb{E}(\boldsymbol{Ax}), \text{ and } Var(\boldsymbol{Ax}) \text{ to find } p(\boldsymbol{y}): \\
& \quad  \quad p(\boldsymbol y) = \mathcal{N} (\boldsymbol y | A \boldsymbol\mu, \boldsymbol A \boldsymbol \Sigma \boldsymbol A^T).
\end{align}
$$

$$
section 7b\\
\begin{align}

Exponential(x; \lambda)& = p(x ; \lambda )  = \begin{cases}
\lambda e^{- \lambda x} & x \ge 0,\\
0 & x < 0.
\end{cases}
\\
Y & = \sum_{j=1}^k X_j, \quad X_j ∼ Exponential(x_j; \lambda) = Gamma(1, \lambda) \\
Y &  ∼ Gamma(k, \lambda)
\\
Y &  ∼ Gamma(\alpha, \beta), \quad \text{$\alpha=k$ is the shape parameter and $\beta$ is the rate parameter}, or \\
Y &  ∼ Gamma(k, \theta), \quad \text{ $k$ is the shape parameter and $\theta= \frac{1}{\beta}$ is the scale.} \\


\end{align}
$$

$$
section 8 \\
\text{If $X_1, \cdots, X_n$ are independent Bernoulli-distributed random variables with probability $\theta$, } \\
\begin{align}
P(X=x)
  &  = \theta^{x_1} (1-\theta)^{x_1} \theta^{x_2} (1-\theta)^{x_2} \cdots \theta^{x_n} (1-\theta)^{x_n} \\
  & = \theta^{\sum_j x_j} (1-\theta)^{n-\sum_j x_j} \\
  & = h(x) g(\theta, T(x)), \\
  & \text{ where $h(x)=1$ and $g(\theta, T(x)) = \theta^{T(x)} (1-\theta)^{n - T(x)}$ and $T(x)=\sum_j x_j$}\\
P(X=x) & = P(X_1=x_1, X_2 = x_2, \cdots, X_n = x_n)\\
& = \frac{e^{-\theta}\theta^{x_1}}{x_1!} \cdot \frac{e^{-\theta}\theta^{x_2}}{x_2!} \cdots \frac{e^{-\theta}\theta^{x_n}}{x_n!}, \\
& = \frac{1}{x_1! x_2! \cdots x_n!} e^{-n \theta}  \theta^{\sum_j x_j} \\
& = h(x) g(\theta, T(x)), \\
&\text{ where } h(x) = \frac{1}{x_1! x_2! \cdots x_n!},   g_\theta(T(x)) = e^{-n \theta}  \theta^{\sum_j x_j}  \text{, and } \\
& \quad \quad \quad T(x) = \sum_j x_j. \\
P(X) & =  h(x) g_\theta(T(x)), \\
p(x | \theta) &= \frac{1}{\sqrt{2 \pi \sigma^2}} exp({- \frac{(x -\mu)^2}{2 \sigma^2})} \\
&= \frac{1}{\sqrt{2 \pi}} exp(\theta^T T(x) - \log \sigma - \frac{\mu^2}{2 \sigma^2}) \\
& \text{where } h(x) =  \frac{1}{\sqrt{2 \pi}},  T(x) = \begin{pmatrix}
x  \\
x^2
\end{pmatrix},  \theta =
\begin{pmatrix}
\mu/\sigma^2  \\
-1 /(2 \sigma^2)
\end{pmatrix} \text{ and } \\
& \quad \quad \quad A(\theta) = \log \sigma + \frac{\mu^2}{2 \sigma^2}. \\
η(\theta) = log(\frac{\theta}{1-\theta})  \quad & ⇒ \quad  \theta(η)  = \frac{1}{1 + e^{-η}} \quad \text{(logistic function/sigmoid).}
\\
f(k, t; \lambda ) & = \frac{(\lambda t)^{k}e^{-\lambda t}}{k!}. \\
F(t) & = P(T ≤ t)  = 1 - P(T>t),\\
& = 1 -  \Big(P(0, T=t) + P(1, T=t) + \cdots + P(k-1, T=t) \Big)\\
& = 1 - \sum_{x=0}^{k-1} \frac{(\lambda t)^{x}e^{-\lambda t}}{x!}. \\
\text{pdf} = \frac{d F(t)}{dt} &= \frac{d}{dt} \Big(1 - \sum_{x=0}^{k-1} \frac{(\lambda t)^{x}e^{-\lambda t}}{x!}\Big), \\
& = \lambda e^{-\lambda t} + \lambda e^{-\lambda t} \sum_{x=1}^{k-1}  \Big(\frac{(\lambda t)^x}{x!} - \frac{(\lambda t)^{x-1}}{(x-1)!} \Big),\\
& =  \lambda e^{-\lambda t} + \lambda e^{-\lambda t} \Big(\frac{(\lambda t)^{k-1}}{(k-1)! } - 1 \Big),\\
& =  \frac{\lambda e^{-\lambda t} (\lambda t)^{k-1}}{(k-1)! },\\
& =  \frac{\lambda^k t^{k-1} e^{-\lambda t} }{\Gamma(k), }\\
& =  Gamma(k, \lambda). \\

\end{align}
$$

$$
section 9 \\
\begin{align}

\overline{X} &= \frac{X_1 + X_2 + \cdots + X_n}{n}.\\
Var(\overline{X}) &= Var( \frac{X_1 + X_2 + \cdots + X_n}{n}) \\
& = \frac{1}{n} \sigma^2. \\

\\
\overline{X} &= \frac{X_1 + X_2 + \cdots + X_n}{n}.\\
Var(\overline{X}) &= Var( \frac{X_1 + X_2 + \cdots + X_n}{n}) \\ &= \frac{1}{n^2} Var( X_1 + X_2 + \cdots + X_n) \\
& = \frac{n}{n^2} Var(X_1) \\
& = \frac{1}{n} \sigma^2, \\
SD(\overline{X}) &= \frac{\sigma}{\sqrt{n}}. \\
& \frac{1}{\sqrt{n}} \quad \text{ or } \quad \sqrt{\frac{2}{{n}}} \text{ for ReLU}. \\
\overline{X} &= \frac{1}{n} \sum^n_{i=1} X_i. \quad X_i  ∼ Distribution(\mu, \sigma^2).
\\
\overline{X} & = \frac{1}{n} \sum^n_{i=1} X_i. \quad X_i  ∼ Distribution(\mu, \sigma^2). \\   
\overline{X} & ∼ \mathcal{N} (\mu_{\overline{X}},\sigma^2_\overline{X}) = \mathcal{N}  (\mu, \frac{\sigma^2}{n}). \\
- 1.96 \sigma_\overline{X} \\
& \quad \quad \quad CI  =  \overline{X} ± z \frac{\sigma}{\sqrt{n}}. \\
\\
&  \overline{X} - z \frac{\sigma}{\sqrt{n}} \quad (\text{lower bound}).\\
\\
 CI &   =
\overline{X} ± t \cdot \text{SE} = \overline{X} ± \text{ME} =\overline{X} +
t \frac{s}{\sqrt{n}}, \\
 \text{ME} & = t \frac{s}{\sqrt{n}}.\\

 n & = \Big(\frac{t \cdot s}{\text{ME}}\Big)^2 ≈
 \Big(\frac{z \cdot s}{\text{ME}}\Big)^2.
\\
n^{*} & = \frac{n}{1 - \text{dropoff rate}}.\\

& \quad \quad \quad CI  = \overline{X} +
t \frac{s}{\sqrt{n}}   . \\
\\
& H_0: \mu_{drug} = \mu_{no\_drug} - \text{ margin}, \\
& H_1: \mu_{drug} < \mu_{no\_drug} - \text{ margin}. \\
\\\
& H_0: \mu_{drug} = \mu_{no\_drug} - 10 = 130, \\
& H_1: \mu_{drug} <  130. \\
& H_0: \mu_{drug} = 115, \\
& H_1: \mu_{drug} < 115 . \\
\\
& H_0: \mu_{drug} = \mu_{no\_drug} = 140, \\
& H_1: \mu_{drug} <  140. \\
\\
\\
& S = 24, n = 36, \text{SE} = \frac{24}{\sqrt{36}} = 4, \\
& H_0: \mu = 140, \\
& H_1: \mu < 140. \\
\\
& \text{Under } H_0: \mu_{0} = 140, \\
& \text{Under } H_1: \mu_{1} = 132. \\
& H_1: \mu_{drug} < 115 . \\
\\

& \overline{X} ± t \frac{S}{\sqrt{n}}. \\
CI & =  \overline{X} ± 1.96 \sigma_\overline{X}, \text{ where }  \sigma_\overline{X} = \frac{\sigma}{\sqrt{n}} \text{ is the SD of the sample mean.}\\
\\
& p(CI_{low} ≤ \mu ≤ CI_{hi})  = 0 \text{ or } 1. \\
\\
\\
\overline{X} &= \frac{1}{n} \sum^n_{i=1} X_i, \quad S^2  = \frac{1}{n-1} \sum_{i=1}^n (X_i - \overline{X} )^2,
\quad
 X_i \sim \mathcal{N}(μ, σ² ), \quad
T = \frac{\overline{X} - \mu}{s/\sqrt{n}}. \\
\boldsymbol{x}^T\boldsymbol{M}\boldsymbol{x} & > 0 \text{ except } \boldsymbol{x} = 0\\
\boldsymbol{M} \boldsymbol{x} & = \lambda \boldsymbol{x} \\
\boldsymbol{x}^T \boldsymbol{M} \boldsymbol{x} &  = \boldsymbol{x}^T \lambda \boldsymbol{x} = \lambda \boldsymbol{x}^T  \boldsymbol{x} = \lambda ‖\boldsymbol{x} ‖^2 > 0, \text{ excep } x =0
\\
\boldsymbol{M} & = \boldsymbol{A}^T \boldsymbol{A} \\
\boldsymbol{x}^T \boldsymbol{M} \boldsymbol{x} &
 = \boldsymbol{x}^T \boldsymbol{A}^T \boldsymbol{A} \boldsymbol{x}
 = ( \boldsymbol{A} \boldsymbol{x})^T (\boldsymbol{A} \boldsymbol{x} )
 = ‖\boldsymbol{A} \boldsymbol{x} ‖^2 ≥ 0
\\
H_0 &: \mu_1 = \mu_2, \quad  \quad \quad \quad H_0 : \mu_1 = \mu_2, \\
H_1 &: \mu_1 ≠ \mu_2. \quad \text{ or } \quad H_1: \mu_1 < \mu_2. \\

\text{t-score} &= \ \frac{\overline{x} - \mu_0}{ s/\sqrt{n}},\\

\text{t-score} &= \ \frac{135-140}{ 12/\sqrt{36}} = -2.5.
\\
& \text{p-value } = p(\text{z-score}=2.5 | h_0 \text{ is true}) = p(\text{z-score}=2.5 | h_0 \text{ is true})
\\
\text{t-score} &= \ \frac{\overline{x} - \mu}{ s/\sqrt{n}}
, \quad
\text{z-score} = \ \frac{\overline{x} - \mu}{ \sigma/\sqrt{n}}, \quad \text{ where $\overline{x}$ is the sample mean,}
\\
\\

\text{z-score} & = \ \frac{\overline{x} - \mu}{ \sigma/\sqrt{n}}, \quad \text{ where $\overline{x}$ is the sample mean, $\mu$ and $\sigma^2$ are the population mean and variance.}
\\

\overline{x} &= \mu_1, \quad \mu = \mu_2. \\
\\
\end{align} \\

$$

$$
section 10\\
\begin{align} \\
H_0 &: \mu_1 = \mu_2,  \quad & H_0 : \mu_1 = \mu_2,  \quad \quad   \quad & H_0 : \mu_1 = \mu_2, \\
H_1 &: \mu_1 ≠ \mu_2.  \quad & H_1 : \mu_1 < \mu_2.  \quad \quad   \quad & H_1 : \mu_1 > \mu_2. \\
\\

\end{align} \\
$$

$$
section 11\\
\begin{align} \\
\sigma_{\overline{X}} = \frac{\sigma}{\sqrt{n}} & \rightarrow \sigma_{\overline{X}} = \frac{s}{\sqrt{n}}
\\
\overline{X^m}
\\
\text{Type I error } & = P(\text{Reject } H_0| H_0 \text{ is true)} =\alpha.\\

\text{Type I error } & = P(\text{Reject } H_0| H_0 \text{ is true)},\\
   \text{Type II error } & = P(\text{Fail to reject } H_0| H_1 \text{ is true}).\\

\beta &= P(\overline{X}>134 | \mu_1 = 132) \\
\text{z-score} &= \frac{134-132}{24/\sqrt{36}} = 0.5 \\
\text{p-value} &= 0.31 \text{ (for one-sided testing)},\\
\text{power} &= 0.69. \\
H_0 &: \mu = \mu_0 \\
H_1 &: \mu > \mu_0 \\
& \text{reject $H_0$ if p-value }  ≤ \alpha, \\
& \text{fail to reject $H_0$ if p-value }  > \alpha. \\
CI &= {\overline{X}} ± ME = {\overline{X}} ± z \cdot SE \quad \text{where SE is the SD of } \overline{X}.\\
& H_0: \mu_A  = \mu_B, \\
& H_1: \mu_A  > \mu_B, \\
&\text{ where $\mu_A$ and $\mu_B$ are the sample mean after and before taking the course respectively.}\\
& H_0: \mu_A - \mu_B = 0, \\
& H_1: \mu_A  - \mu_B > 0. \\
& \quad \quad or \\
& H_0: \mu_d  = 0, \\
& H_1: \mu_d  > 0. \\

& \text{t-score} = \frac{{\overline{d} - \mu_{d}}}{\text{SE}/\sqrt{n}} = \frac{74-0}{13.2/\sqrt{n}}. \\
& CI = \overline{d} ± t \cdot \text{SE} = 74 ± t \cdot 13.2/\sqrt{n} \\
& H_0: \text{Median}_{diff} = 0 \\
& H_1: \text{Median}_{diff} > 0. \\
& p(sum+ = 41.5 | \text{ expected sum+ } = 27.5) \\
& \quad \text{ where sum of rank is } \sum_1^{10} i = 55, \text{ and the expected sum of rank is } 55/2 = 27.5. \\
& \text{Group a}: \overline{y}_a = 100,  \quad S_a = 10, \quad n_a = 16, \\
& \text{Group b}: \overline{y}_b = 120,  \quad S_b = 15, \quad n_b = 25. \\
& H_0: \overline{y}_a - \overline{y}_b = 0, \\
& H_1: \overline{y}_a - \overline{y}_b ≠ 0, \\
Var(\overline{y}_a - \overline{y}_b) & = Var(\overline{y}_a) + Var(\overline{y}_b) \quad \text{if $\overline{y}_a$ and $\overline{y}_b$ are independent.} \\
&= \frac{S_a^2}{n_a} +  \frac{S_b^2}{n_b}. \\

\text{SE}_{\overline{y}_a - \overline{y}_b} & = \sqrt{Var(\overline{y}_a - \overline{y}_b) } \\
& = \sqrt{\frac{10^2}{16} + \frac{15^2}{25}}. \\

& \text{t-score}  = \frac{(100-120)-0}{\text{SE}_{\overline{y}_a - \overline{y}_b}}.\\
S^2_{\text{pooled}} &= \frac{(n_a-1)S^2_{a} + (n_b-1)S^2_{b}}{n_a-1 + n_b-1} \\
\text{SE} &= \sqrt{\frac{S^2_{\text{pooled}}}{n_a} + \frac{S^2_{\text{pooled}}}{n_b}}.\\
n_a + n_b - 2.\\
& \text{p-value} = \frac{\text{num of } (\text{diff}_{boostrap} > \text{diff}_{observed}) }{b}.
\\
& \text{p-value} = \frac{\text{num of } (\text{diff}_{boostrap} > \text{diff}_{observed}) }{p}.
\\
& \overline{y}_a  - \overline{y}_b \\
&X_{g_{1_1}} \\
& \mu_{g_3} \\
& SS = \sum_i^n (X_i -\mu_X )^2. \\
& SS_{\text{unexpl}} = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{i_j} -\mu_{g_i} )^2 = \sum_\text{all} (X_{i_j} -\mu_{g_i} )^2  \\
\\
& \quad \text{where $k$ is the number of group}, \\
& \quad \quad \quad \quad \text{$\mu_{g_i}$ is the sample mean for group $i$}, \\
& \quad \quad \quad \quad \text{$n_i$ is the number of data points in group $i$}.
\\
& SS_{\text{expl}} = \sum_{i=1}^k \sum_{j=1}^{n_i} (\mu_{g_i} - \mu )^2 =  \sum_{i=1}^k n_i (\mu_{g_i} - \mu)^2.  \\

& SS_{\text{expl}} = \sum_1^k (\mu_{g_{i} - \mu})^2 \\

\\
& SS_{\text{total}} = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{i_j} -\mu )^2 = \sum_\text{all} (X_{i_j} -\mu )^2 . \\
\\
& SS_{\text{total}} = SS_{\text{explain}} + SS_{\text{unexpl}}.\\
& SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}. \\
& \text{MS} = \frac{\text{SS}}{\text{df}}. \\
\text{F-ratio} & = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} \\
&=
\frac{\text{SS}_{\text{between}}/\text{df}_{\text{between}}}{\text{SS}_{\text{within}}/\text{df}_{\text{within}}}\\
&= \frac{\text{SS}_{\text{between}}/(k-1)}{\text{SS}_{\text{within}}/(n-k)}. \\
.
\\
& H_0: \mu_1 = \mu_2, \\
& H_1: \mu_1 \neq \mu_2, \\
& \text{t-score} = \frac{100 - 120}{SE}


\\
& H_0: \mu_1 = \mu_2 = \mu_3  \\
& H_1: \text{means are not all equal.} \\
&\mu = \frac{\sum_{i=1}^k \sum_{j=1}^{n_k} X_{i_j}}{count } =  \frac{\sum_{all} X_{i_j}}{count} \quad \text{ where } count = \sum_{j=1}^k {n_j} \text{ and } n_j \text{is the sample size for sample } j. \\
& U = \overline{y}_a  - \overline{y}_b. \\
& Q = \sum^k_1 Z_i^2, \quad \text{where } Z_i ~ \sim \mathcal{N}(0, 1) \\
& Q \sim \chi^2(k) \text{ or } \chi^2_k. \\
& U = \frac{S_1/d_1}{S_2/d_2} \quad \text{ where } S_1, S_2 \sim \chi^2, \\
& \sum_{i=1}^n X_i = 0, \quad S^2 = \frac{1}{n-1} \sum_{i=1}^n X_i^2

\\
& \text{numerator degrees of freedom $\nu_1$} = \text{df}_\text{between} = k -1, \\
& \text{denominator degrees of freedom $\nu_2$} = \text{df}_\text{within} = n - k. \\
& SS_\text{total} = SS_A + SS_B + SS_{A \times B} + SS_{\text{within}}. \\

& SS_{A \times B} = SS_\text{between} - SS_A - SS_B. \\



& SS_{A \times B} = SS_\text{total} - SS_A - SS_B - SS_{\text{within}}. \\
& \alpha \rightarrow \frac{\alpha}{t} \quad \text{where $t$ is the number of tests}. \\
& (1-\frac{\alpha}{t})^t ≈ (1-\alpha). \\
& H_0: p_{{cat}_{male}} = p_{{cat}_{female}}, \\
& H_1: p_{{cat}_{male}} \ne p_{{cat}_{female}}. \\
p(\text{male } \& \text{ cat}) & = p(\text{male}) p(\text{cat}) \text{ if they are independent.} \\
& = \frac{700}{1500}\frac{600}{1500}.  \\
\# \text{ cat adopted by male } &=\frac{700}{1500}\frac{600}{1500} 1500, \\
&=\frac{700 × 600}{1500}. \\
\text{count in $cell_{ij}$} &= \frac{R_i × C_j}{T}.\\
\chi^2 & = \sum_{\text{all cell i}} \frac{(O_i -E_i)^2}{E_i}.\\
P(D| E) &= \frac{20}{100} = 0.2, \\
& P(D|\text{ not } E) = \frac{10}{100} = 0.1. \\
\text{risk difference (RD)/attributable risk (AR)} & =
P(D| E) - P(D|\text{ not } E) = 0.2 - 0.1 = 0.1.\\
\text{relative risk (RR)} & = \frac{P(D| E)}{P(D|\text{ not } E)} = \frac{0.2}{0.1} = 2.\\
\text{odds ratio (OR)}&  = \frac{\text{odds($D|E$)}}{ \text{odds($D|$ not $E$})} = \frac{P(D| E)/P(\text{ not $D$}| E)}{P(D| \text{ not $E$})/P(\text{ not $D$}| \text{ not $E$})}, \\
& = \frac{0.2/0.8}{0.1/0.9} = 2.25.\\
\text{odds ratio (OR)} & = \frac{\text{odds($D|E$)}}{ \text{odds($D|$ not $E$})} =
\frac{\text{odds($E|D$)}}{ \text{odds($E$ | not $D$)}}.\\
SS_{\text{total}} = SS_{\text{unexplained}} + SS_{\text{explained}},\\

\end{align}
$$

$$
section 12\\
\\
\begin{align}
p(x) & = \frac{\begin{pmatrix}
K \\
x
\end{pmatrix}
\begin{pmatrix}
N - K \\
r -x
\end{pmatrix}}
{
\begin{pmatrix}
N  \\
r
\end{pmatrix}
}.
\\
\\
p(2) & = \frac{\begin{pmatrix}
9 \\
2
\end{pmatrix}
\begin{pmatrix}
21 - 9 \\
12 - 2
\end{pmatrix}}
{
\begin{pmatrix}
21  \\
12
\end{pmatrix}
}.
\\
y & = f_1(x_*^{(1)}) \\ \\
& \boldsymbol D =  \{(x^{(i)}, y^{(i)})\}^{m}_{i=1}. \\
\\
y^{(i)} &= \theta^T x^{(i)} + \epsilon^{(i)}, \quad \quad i = 1 \dots m. \\
θ & \sim \mathcal{N}\Big(0, \tau^2 I \Big), \quad \quad \theta \in ℝ^n,
\\
\epsilon &\sim \mathcal{N}\Big(0, σ² \Big), \quad \quad  \epsilon \in ℝ^m.\\
\\
p(\boldsymbol y_{*} | \boldsymbol x_{*}, \boldsymbol D)
&= \int_\boldsymbol \theta p(\boldsymbol y_{*} | \boldsymbol x_{*}, \boldsymbol \theta) p(\boldsymbol \theta | \boldsymbol D) d \boldsymbol \theta.
\\
\quad \text{ where $\boldsymbol D$ is the training dataset}.\\
& X \in ℝ^{m \times n}, \quad  y \in ℝ^m
\\
P\Big(y^{(i)} | x^{(i)}, \theta \Big)  & = \frac{1}{\sqrt{2 \pi } \sigma}   \exp \Big( - \frac{ (y^{(i)} -\theta^T x^{(i)}  )^2}{2\sigma^2}\Big). \\
\theta | D & \sim \mathcal{N}\Big( \frac{1}{\sigma^2}A^{-1}X^Ty , A^{-1} \Big) \quad \text{where } A = \frac{1}{\sigma^2}X^TX + \frac{1}{\tau^2}I. \\
y_* | x_{*}, D & \sim \mathcal{N}\Big( \frac{1}{\sigma^2}x^T_*A^{-1}X^Ty, x^T_* A^{-1}x_* + \sigma^2 \Big).\\
P(Y= y_* | x_* , D). \\
\\   
&\begin{bmatrix}
f(\mbox x_1) \\
f(\mbox x_2)\\
\vdots \\
f(\mbox x_{12}) \\
\end{bmatrix} \sim
\mathcal{N}
\begin{pmatrix}
  \begin{bmatrix}
  m(\mbox x_1) \\
  m(\mbox x_2)\\
  \vdots \\
  m(\mbox x_{12})\\
  \end{bmatrix},
  \begin{bmatrix}
  k(\mbox x_1, \mbox x_1) &  k(\mbox x_1, \mbox x_2) & k(\mbox x_1, \mbox x_3) & k(\mbox x_1, \mbox x_4)\\
  k(\mbox x_2, \mbox x_1) &  k(\mbox x_2, \mbox x_2) & k(\mbox x_2, \mbox x_3) & k(\mbox x_2, \mbox x_4)\\
  \vdots \\
  k(\mbox x_4, \mbox x_1) &  k(\mbox x_4, \mbox x_2) & k(\mbox x_4, \mbox x_3) & k(\mbox x_4, \mbox x_4)\\
  \end{bmatrix}

\end{pmatrix} . \\
f & \sim GP(*, *),
\\
 \boldsymbol{x} & \sim  \mathcal{N} (\boldsymbol{\mu_x}, \boldsymbol{\Sigma_x}), \quad y \sim  \mathcal{N} (\boldsymbol{\mu_y}, \boldsymbol{\Sigma_y}), \\
& \boldsymbol{x} + \boldsymbol{y} \sim  \mathcal{N} (\boldsymbol{\mu_x} + \boldsymbol{\mu_y}, \boldsymbol{\Sigma_x} + \boldsymbol{\Sigma_y)}. \\
& n = 8 \frac{s^2}{d^2}.
\\
f &\sim \mathcal{G}\mathcal{P}(\cdot, \cdot). \\
&  \rightarrow f^1, f^2, f^3, f^4, \dots \\
\\
\\
p(\boldsymbol \theta | \boldsymbol D ) & = \frac{p(\boldsymbol D | \boldsymbol \theta) p(\boldsymbol \theta)}{\int_\theta p(\boldsymbol D|\boldsymbol \theta)p(\boldsymbol \theta)  d \boldsymbol \theta}
\propto p(\boldsymbol D | \boldsymbol \theta) p(\boldsymbol \theta) = s \mathcal{N}(\boldsymbol \theta | \boldsymbol c, \boldsymbol C)
 \\
 p(\boldsymbol \theta | \boldsymbol D ) & \propto s \mathcal{N}(\boldsymbol \theta | \boldsymbol c, \boldsymbol C)
\\
\\
p(\boldsymbol \theta | \boldsymbol D ) & = \mathcal{N}(\boldsymbol \theta | \boldsymbol c,\boldsymbol  C), \quad \text{ as both are probability distributions.}\\
\\
&p(\theta | D ) \propto p(D | \theta) p(\theta)

\\
&\boldsymbol{\Sigma}_{\boldsymbol{{xy}}_{ij}} = Cov(x_i, x_j).
\\
&\boldsymbol{y} = \boldsymbol{A} \boldsymbol{x}, \quad X \sim  \mathcal{N} (\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\\
\boldsymbol{y} & = \boldsymbol{\theta}^T \boldsymbol{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I), \quad \boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\\
p(\boldsymbol y_*| \boldsymbol x_*, \boldsymbol{\theta}) & \sim \mathcal{N}( \boldsymbol{\theta}^T \boldsymbol x_*, \sigma^2 \boldsymbol I ) , \quad p(\boldsymbol{\theta}|\boldsymbol{D}) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}).
\\
&\boldsymbol{y} = \boldsymbol{A} \boldsymbol{x}, \quad X \sim  \mathcal{N} (\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\quad
 p(\boldsymbol y ) = \mathcal{N} (\boldsymbol y | \boldsymbol A \boldsymbol \mu, \boldsymbol{A \Sigma A}^T ).
\\
\\
&\boldsymbol{y} =  \boldsymbol{\theta}^T \boldsymbol{x_*}, \quad \boldsymbol \theta \sim  \mathcal{N} (\boldsymbol{\mu}, \boldsymbol{\Sigma}),
\quad
 p(\boldsymbol y ) = \mathcal{N} (\boldsymbol y | \boldsymbol{\mu}^T \boldsymbol x_*, \boldsymbol{x_*^T \Sigma x_*} ). \\
\\
p(\boldsymbol y_*| \boldsymbol x_*, \boldsymbol{\theta}) & \sim \mathcal{N}( \boldsymbol \mu_*, \boldsymbol \Sigma_*).
\\
\boldsymbol \mu_* & = \boldsymbol{\mu}^T \boldsymbol x_* , \\
\boldsymbol \Sigma_* & = \boldsymbol{x_*^T \Sigma x_*} + \sigma^2 I.
\\ & \quad \quad \text{where } \boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad
\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I). \\
\\
y & = \theta x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2), \quad \theta \sim \mathcal{N}(\mu, \Sigma).
\\
p( y_*|  x_*, \theta) & \sim \mathcal{N}(  \mu_*,  \Sigma_*).
\\
\mu_* & = \mu x_* , \\
\Sigma_* & = x_*^T \Sigma x_* + \sigma^2 I.
\\ & \quad \quad \text{where } \theta \sim \mathcal{N}(\mu, \Sigma), \quad
\epsilon \sim \mathcal{N}(0, \sigma^2 I). \\
\\
& \mathcal{N} ( x |  a,  A)  \mathcal{N} ( x | b,   B)  = s \mathcal{N} ( x |  c, C),  \\
& \text{where }\\
& C  = ( A^{-1} +  B^{-1})^{-1}, \\
& c  = C ( A^{-1}  a + B^{-1}  b),  \\
& s = (2 \pi)^{-\frac{D}{2}} |  A + B|^{-\frac{1}{2}} \exp(-\frac{1}{2} ( a -  b)^T ( A +  B)^{-1}( a - b)). \\
p( y |  x_*, \theta) & \sim \mathcal{N}(  \mu_*,  \Sigma_*).
\\
\mathcal{N} \Big(
  \Big).

\end{align}
$$

$$
Section 13
\\
\begin{align}

\\
\begin{bmatrix}
f_{x_1}  \\
f_{x_2}  \\
f_{x_3}  \\
\vdots \\
f_{x_{18}}  \\
f_{x_{19}}  \\
f_{x_{20}}  \\
\end{bmatrix} & \sim
\mathcal{N} \Big(
  0, \Sigma
  \Big), \quad \text{where $\Sigma$ is a $20 \times 20$ matrix with } \Sigma_{ij} = s^2 \exp \Big(- \frac{1}{2l^2} (x_i - x_j)^2 \Big).
\\
\\
\begin{bmatrix}
f_{x_1}  \\
f_{x_2}  \\
\vdots \\
f_{x_{599}}  \\
f_{x_{600}}  \\
\end{bmatrix} & \sim
\mathcal{N} \Big(
  0, \Sigma
  \Big), \quad \text{where $\Sigma$' is a $600 \times 600$ matrix}. \quad  
  \begin{bmatrix}
  f_1  \\
  f_2  \\
  \vdots \\
  f_{599}  \\
  f_{600}  \\
  \end{bmatrix} =
  \begin{bmatrix}
    \\
  f  \\
  \\
  \end{bmatrix}
.
\\
\begin{bmatrix}
f_1  \\
f_4  \\
f_7  \\
f_{10}  \\
f_{*_{1}}  \\
f_{*_{2}}  \\
f_{*_{3}} \\
\vdots \\
f_{*_{12}}  \\
\end{bmatrix} & \sim
\mathcal{N} \Big(
  0, \Sigma''
  \Big) \quad \rightarrow \quad
  \begin{bmatrix}
  -0.44  \\
  -0.2  \\
  0.7  \\
  0.5  \\
  f_{*_{1}}  \\
  f_{*_{2}}  \\
  f_{*_{3}}  \\
  \vdots \\
  f_{*_{12}}  \\
  \end{bmatrix}
  \sim
  \mathcal{N} \Big(
    0, \Sigma''
    \Big)
    \quad \rightarrow \quad
    \begin{bmatrix}
    f  \\
    f_*
    \end{bmatrix}
    \sim
    \mathcal{N} \Big(
      0, \Sigma''
      \Big), \quad \text{(unit in million).}\\
& p(f_* |  D ) \text{ where $D$ contains the four quarter reports.}   \\
& \begin{bmatrix}
-0.44  \\
-0.2  \\
0.7  \\
0.5  \\
f_{*_{60}}  \\
f_{*_{280}}  \\
\end{bmatrix}
\sim
\mathcal{N} \Big(
  0, \Sigma''
  \Big). \\
\\
  &  \quad \quad f(\mbox{x}) \sim \mathcal{GP}(m(\mbox{x}), \kappa(\mbox{x}, \mbox{x}')), \\
  \\
  &  \quad \quad \text{where }\\
  & \quad \quad \quad \quad m(\mbox{x}) = \mathbb{E}[f(\mbox{x})], \\
  & \quad \quad \quad \quad \kappa(\mbox{x}, \mbox{x}') = \mathbb{E}[(f(\mbox{x})-m(\mbox{x}))(f(\mbox{x'})-m(\mbox{x'}))^T]. \\
  \\
  &  \quad \quad f(X) \sim \mathcal{N}(m(X), k(X, X)). \\

  & \quad \quad \quad \quad \\
& f^1 = [f^1(\mbox x_1)=0.5, f^1(\mbox x_2)=0.65, ..., f^1(\mbox x_{12})=-1.5], \\
& f^2 = [f^2(\mbox x_1)=-1, f^2(\mbox x_2)=-1.1, ..., f^2(\mbox x_{12})=0.66]. \\

& k_{ij} = \exp \Big(- \frac{1}{2l^2} ‖ x_i - x_j‖^2 \Big).
\\
& X = [\mbox x_1,\mbox x_2, \mbox x_3, \dots , \mbox x_{12}],
\\
\\
\begin{bmatrix}
x_1  \\
x_2  
\end{bmatrix}
& \sim
\mathcal{N} \Big(\begin{bmatrix}
0  \\
3  
\end{bmatrix} ,
\begin{bmatrix}
7 & 2  \\
2 & 1
\end{bmatrix}
  \Big) = p(x_1 , x_2),
\\
\\
\mu_{x_1 | x_2=-2} & =  \mu_x+{\sigma^2}_{x_1x_2}{{\sigma^2}_{x_2x_2}}^{-1}({ x_2}-\mu_{x_2}) = 0 + 2 \cdot 1^{-1} (-2 - 3) = -10, \\
\sigma^2_{x_1 | x_2=-2} & = \sigma^2_{x_1x_1}-\sigma^2_{x_1x_2}{\sigma^2_{x_2x_2}}^{-1}\sigma^2_{x_2x_1}= 7 - 2 \cdot 1^{-1} \cdot 2 = 3. \\
\\

\begin{bmatrix}
x_1  \\
x_2=2 \\
\end{bmatrix}
& \sim
\mathcal{N} \Big(\begin{bmatrix}
0  \\
3  
\end{bmatrix} ,
\begin{bmatrix}
7 & 2  \\
2 & 1
\end{bmatrix}
  \Big) = p(x_1, x_2).
\\
  x_1 | x_2=2  
  & \sim
  \mathcal{N} (-10 , 3
    ) .  
    \\


        \\   
        \\ p(\theta | D ) &=
        \frac{1}{Z} p( D| \theta) p(\theta) \\
        & = \frac{1}{Z} \mathcal{N}( y | {X \theta}, \sigma^2 I) \mathcal{N}(0, \tau^2 I  ) \\
          & \propto
        \mathcal{N}( y | {X \theta}, \sigma^2 I) \mathcal{N}(0, \tau^2 I  ) \\
        & \propto
        \mathcal{N}( \theta |  \mu_{\theta | D},  \Sigma_{\theta | D} ).

        \\
\\
         p( \theta |  D ) & = \mathcal{N}( \theta |  \mu_{\theta | D},  \Sigma_{\theta | D} ).
\\  
\\
    p(\theta | D ) &
    \propto p( D |  \theta) p( \theta) \\
    & = p( y |  X,  \theta) p(\theta) \\
    & = \mathcal{N}( y | X \theta, \sigma^2I)
    \mathcal{N}(\theta | 0, \tau^2 I ) \\
    & \propto \exp \Big(
      -\frac{1}{2\sigma^2} (y-X\theta)^T(y-X\theta)
  \Big) \exp \Big(   -\frac{1}{2 \tau^2} \theta^T\theta\Big) \\
  & = \exp \Bigg(
    -\frac{1}{2} \Big[ \frac{1}{\sigma^2} (y-X\theta)^T(y-X\theta) + \frac{1}{\tau^2} \theta^T\theta \Big]
\Bigg) \\
& = \exp \Bigg(
  -\frac{1}{2} \Big[ \frac{1}{\sigma^2} (
 \theta^TX^TX\theta - 2 \theta^TX^Ty + y^Ty)

   + \frac{1}{\tau^2} \theta^T\theta \Big] \Bigg) \\

 & = \exp \Bigg(
    -\frac{1}{2}

    \Big[
      \theta^T \Big( \frac{1}{\sigma^2} X^TX +  \frac{1}{\tau^2} \Big) \theta - 2\Big( \frac{1}{\sigma^2} \Big) y^T X \theta

 \Big] \Bigg) \\
 & = \exp \Bigg(
    -\frac{1}{2} \Big[
      \theta^T A \theta - 2b^T\theta
    \Big]
 \Bigg)  \text{where } A = \frac{1}{\sigma^2}X^TX + \frac{1}{\tau^2}I, b = \frac{1}{\sigma^2} X^Ty \\
 & = \exp \Bigg(
    -\frac{1}{2} \Big[
      (\theta - A^{-1} b)^T A (\theta - A^{-1}b) -
      b^T A^{-1}b
    \Big] \Bigg)\\
  & \propto   \exp \Bigg(
     -\frac{1}{2} \Big[
       (\theta - A^{-1} b)^T A (\theta - A^{-1}b)
     \Big]

 \Bigg) \\
 p(\theta | D ) & = \mathcal{N}\Big(\theta | A^{-1}b, A^{-1} \Big).
\\
\theta | D & \sim \mathcal{N}\Big( \frac{1}{\sigma^2}A^{-1}X^Ty , A^{-1} \Big) \quad \text{where } A = \frac{1}{\sigma^2}X^TX + \frac{1}{\tau^2}I. \\





\end{align}


$$


$$
section 14\\
\\
P(y_*\mid D,\mathbf{x}) \sim \mathcal{N}(\mu_{y_*\mid D}, \Sigma_{y_*\mid D}), \\
\mu_{y_*\mid D} = K_*^T (K+\sigma^2 I)^{-1} y \\
\Sigma_{y_*\mid D} = K_{**} - K_*^T (K+\sigma^2 I)^{-1}K_*.
\\
\\
\text{Prior: } P(f\mid \mathbf{x}) \sim \mathcal{N}(\mu, \Sigma) \\
k_{ij} = s^2 \exp \Big(- \frac{1}{2l^2} ‖x_i - x_j‖^2 \Big).\\

\\
\begin{bmatrix}
f(\mbox x_1) \\
f(\mbox x_2)\\
\vdots \\
f(\mbox x_{12}) \\
\end{bmatrix} \sim
\mathcal{N}
\begin{pmatrix}
  \begin{bmatrix}
  m(\mbox x_1) \\
  m(\mbox x_2)\\
  \vdots \\
  m(\mbox x_{12})\\
  \end{bmatrix},
  \begin{bmatrix}
  k(\mbox x_1, \mbox x_1) &  k(\mbox x_1, \mbox x_2) & \cdots & k(\mbox x_1, \mbox x_{12})\\
  k(\mbox x_2, \mbox x_1) &  k(\mbox x_2, \mbox x_2) & \cdots & k(\mbox x_2, \mbox x_{12})\\
  \vdots \\
  k(\mbox x_{12}, \mbox x_1) &  k(\mbox x_{12}, \mbox x_2) & \cdots & k(\mbox x_{12}, \mbox x_{12})\\
  \end{bmatrix}

\end{pmatrix} . \\

\begin{bmatrix}
f(\mbox x_1) \\
f(\mbox x_2)\\
\vdots \\
f(\mbox x_{12}) \\
\end{bmatrix} \sim
\mathcal{N}
\begin{pmatrix}
  0,
  \begin{bmatrix}
  k(\mbox 1, \mbox 1) &  k(\mbox 1, 2) & \cdots & k(\mbox 1, \mbox 12)\\
  k(\mbox 2, \mbox 1) &  k(\mbox 2, 2) & \cdots & k(\mbox 2, \mbox 12)\\
    \vdots \\
    k(\mbox 12, \mbox 1) &  k(\mbox 12, 2) & \cdots & k(\mbox 12, \mbox 12)\\
  \end{bmatrix}
\end{pmatrix} . \\

\begin{bmatrix}
\mbox x_1  \\
\mbox x_2  
\end{bmatrix}
\sim
\mathcal{N} \Big(\begin{bmatrix}
\mu_1  \\
\mu_2  
\end{bmatrix} ,
\begin{bmatrix}
\Sigma_{11} & \Sigma_{12}  \\
\Sigma_{21} & \Sigma_{22}
\end{bmatrix}
  \Big).
\\
  \begin{align*} {\rm Cov}({\mbox z}, {\mbox x}_2) &=
  {\rm Cov}({\mbox x}_1 + A {\mbox x}_2 , {\mbox x}_2)\\
  &={\rm Cov}( {\mbox x}_{1}, {\mbox x}_2 ) +
  {\rm Cov}({A}{\mbox x}_2, {\mbox x}_2) \\
  &= \Sigma_{12} + {A} {\rm Var}({\mbox x}_2) \\
  &= \Sigma_{12} - \Sigma_{12} \Sigma^{-1}_{22} \Sigma_{22} \\
  &= 0.
  \end{align*}\\
  \quad \quad \quad \quad A = -\Sigma_{12} \Sigma^{-1}_{22}.
\\
𝔼({\mbox z}) = { \mu}_1 + {A}  { \mu}_2. \\


$$

$$
section 15\\

\begin{align}


𝔼({\mbox x}_1 | {\mbox x}_2) &= 𝔼( {\mbox z} - {A} {\mbox x}_2 | {\mbox x}_2) \\
& = 𝔼({\mbox z}|{\mbox x}_2) -  𝔼({\mbox A}{\mbox x}_2|{\mbox x}_2) \\
& = 𝔼({\mbox z}) - {\mbox A}{\mbox x}_2 \\
& = {\mu}_1 + {A}  ({\mu}_2 - {\mbox x}_2) \\
& = {\mu}_1 + \Sigma_{12} \Sigma^{-1}_{22} ({\mbox x}_2- {\mu}_2). \\

\\
{\rm Var}({\mbox x}_1|{\mbox x}_2) &= {\rm Var}({\mbox z} - {A} {\mbox x}_2 | {\mbox x}_2) \\
&= {\rm Var}({\mbox z}|{\mbox x}_2) + {\rm Var}({A} {\mbox x}_2 | {\mbox x}_2) - {A}{\rm Cov}({\mbox z}, -{\mbox x}_2) - {\rm Cov}({\mbox z}, -{\mbox x}_2) {A}^T \\
&= {\rm Var}({\mbox z}|{\mbox x}_2) + 0 - 0 - 0 \quad \quad \text{since $z$ and $x_2$ are uncorrelated.}\\
&= {\rm Var}({\mbox z}).
\\

{\rm Var}({\mbox x}_1|{\mbox x}_2) & = {\rm Var}( {\mbox z} ) \\
&= {\rm Var}( {\mbox x}_1 + {A} {\mbox x}_2 ) \\
&= {\rm Var}( {\mbox x}_1 ) + {A} {\rm Var}( {\mbox x}_2 ) {A}^T
+ {A} {\rm Cov}({\mbox x}_1,{\mbox x}_2) + {\rm Cov}({\mbox x}_2,{\mbox x}_1) {A}^T \\
&= \Sigma_{11} +\Sigma_{12} \Sigma^{-1}_{22} \Sigma_{22}\Sigma^{-1}_{22}\Sigma_{21}
- 2 \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \\
&= \Sigma_{11} +\Sigma_{12} \Sigma^{-1}_{22}\Sigma_{21}
- 2 \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \\
&= \Sigma_{11} -\Sigma_{12} \Sigma^{-1}_{22}\Sigma_{21}. \\

{\bf x}_1 | {\bf x}_2 & \sim
\mathcal{N} \Big(
\boldsymbol{\mu_1} + \boldsymbol{\Sigma_{12}} \boldsymbol{\Sigma^{-1}_{22}} (\boldsymbol{{x}_2}- \boldsymbol {\mu_2}), \boldsymbol{\Sigma_{11}} - \boldsymbol{\Sigma_{12}} \boldsymbol{\Sigma^{-1}_{22}}\boldsymbol{\Sigma_{21}}
\Big).
\\
& k_{SE}(x, x') = s^2 \exp(-\frac{1}{2 l^2} ‖ x_i - x_j‖^2),
\\

\boldsymbol v^\top K \boldsymbol v & = \frac{1}{n} \boldsymbol v^\top\Big((\boldsymbol x - \boldsymbol {\mu_x})(\boldsymbol x - \boldsymbol{\mu_x})^\top \Big) \boldsymbol v \\
& \propto \Big( \boldsymbol  v^\top(\boldsymbol x - \boldsymbol {\mu_x})\Big) \Big( (\boldsymbol x - \boldsymbol{\mu_x})^\top  \boldsymbol v \Big) \\
& = \Big( (\boldsymbol x - \boldsymbol{\mu_x})^\top  \boldsymbol v \Big)^\top \Big( (\boldsymbol x - \boldsymbol{\mu_x})^\top  \boldsymbol v \Big) \\
& =  ‖ (\boldsymbol x - \boldsymbol{\mu_x})^\top  \boldsymbol v ‖^2  ≥ 0 \quad \text{for any } v.\\
& \boldsymbol v^\top K \boldsymbol v ≥ 0 \text{ for any } v.
\\
&
\begin{bmatrix}
f(\mbox x_1)  \\
f(\mbox x_4)  \\
f(\mbox x_7)  \\
f(\mbox x_{10})  \\
f_*(\mbox x_1)  \\
f_*(\mbox x_{2})  \\
\vdots \\
f_*(\mbox x_{12})  \\
\end{bmatrix}
\sim
\mathcal{N} \Bigg(
  \begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
\vdots\\
0
\end{bmatrix}
,
\begin{bmatrix}
k(1, 1) & k(1, 4)  & k(1, 7) & k(1, 10) &  k(1, 1) & k(1, 2) & \cdots & k(1, 12)   \\
k(4, 1) & k(4, 4)  & k(4, 7) & k(4, 10) &  k(4, 1) & k(4, 2) & \cdots & k(4, 12)   \\
k(7, 1) & k(7, 4)  & k(7, 7) & k(7, 10) &  k(7, 1) & k(7, 2) & \cdots & k(7, 12)   \\
k(10 , 1) & k(10, 4)  & k(10, 7) & k(10, 10) &  k(10, 1) & k(10, 2) & \cdots & k(10, 12)   \\


k(1, 1) & k(1, 4)  & k(1, 7) & k(1, 10) &  k(1, 1) & k(1, 2) & \cdots & k(1, 12)   \\
k(2, 1) & k(2, 4)  & k(2, 7) & k(2, 10) &  k(2, 1) & k(2, 2) & \cdots & k(2, 12)   \\
\vdots \\
k(12, 1) & k(12, 4)  & k(12, 7) & k(12, 10) &  k(12, 1) & k(12, 2) & \cdots & k(12, 12)   \\
\end{bmatrix}
  \Bigg).\\
&\text{ Prior: } p(f_*  | x) ∼ \mathcal{N}(μ, Σ) \\
&\text{ posterior predictive distribution: } p(f_∗∣x_∗,D) \\
\\
&
\begin{bmatrix}
\bf f  \\
\bf f_*  \\
\end{bmatrix}
\sim
\Bigg(
\begin{bmatrix}
\boldsymbol{\mu}  \\
\boldsymbol \mu_*  \\
\end{bmatrix},
\begin{bmatrix}
\boldsymbol K & \boldsymbol {K_{*}}   \\
\boldsymbol {K_*}^\top &  \boldsymbol {K_{**}}  \\
\end{bmatrix}
\Bigg), \\
& p(\mathbf{f}_* \mid , \boldsymbol{x_*},  \boldsymbol{x}, \mathbf{f}) =  \mathcal{N}(\boldsymbol {\mu}_{f_*|x_*, D}, \boldsymbol{\Sigma}_{f_{*}|x_*, D}). \\
\\
& \boldsymbol {\mu}_{f_*|x_*, D} = \boldsymbol {\mu_*} + \boldsymbol {K_{*}}^\top \boldsymbol K^{-1} (\mathbf{f} - \boldsymbol \mu), \\
& \boldsymbol \Sigma_{f_*|x_*, D}  = \boldsymbol K_{**} - \boldsymbol K_{*}^\top \boldsymbol K^{-1} \boldsymbol K_{*}.
\\
&
\begin{bmatrix}
\bf y  \\
\bf y_*  \\
\end{bmatrix} =
\begin{bmatrix}
\bf f  \\
\bf f_*  \\
\end{bmatrix}  +
\begin{bmatrix}
\boldsymbol \epsilon  \\
\boldsymbol \epsilon_*  \\
\end{bmatrix}  
\sim
\Bigg(
\begin{bmatrix}
\boldsymbol{\mu}  \\
\boldsymbol \mu_*  \\
\end{bmatrix},
\begin{bmatrix}
\boldsymbol K + \sigma^2 I & \boldsymbol {K_{*}}   \\
\boldsymbol {K_*}^\top &  \boldsymbol {K_{**}} + \sigma^2 I   \\
\end{bmatrix}
\Bigg). \\

& \boldsymbol {\mu}_{f_*|x_*, D} = \boldsymbol {\mu_*} + \boldsymbol {K_{*}}^\top (\boldsymbol  K + \sigma^2 I) ^{-1} (\mathbf{f} - \boldsymbol \mu), \\
& \boldsymbol \Sigma_{f_*|x_*, D}  = \boldsymbol K_{**} + \sigma^2 I - \boldsymbol K_{*}^\top  (\boldsymbol K + \sigma^2 I)^{-1} \boldsymbol K_{*}.
\\
& \boldsymbol y = \boldsymbol f + \boldsymbol \epsilon, \quad \quad \text{where } \epsilon \sim \mathcal{N} (0, \sigma^2)

\end{align}
$$

$$
\begin{align}
section 15 b\\
\text{Assocative laws: } & a + (b + c)  = (a + b) + c, \\
& a \cdot (b \cdot c)  = (a \cdot b ) \cdot c, \\
\text{Commutative laws: } & a + b  = b + a, \\
& a \cdot b  = b \cdot a, \\
\text{Distributive laws: } &
a \cdot (b + c)  = a \cdot b + a \cdot c, \\
& (a + b) \cdot c  = a \cdot c + b \cdot c. \\
\text{Identity laws: } & a + 0 = 0 + a = a, \\
& 1 \cdot a = a \cdot 1 = a .\\
\text{Inverse laws: } & a+ (-a) = (-a) + a = 0, \\
& a \cdot a^{-1} = a^{-1} \cdot a = 1 \text{ when a ≠ 0.} \\
& ⟨ F, + , \cdot, 0, 1⟩. \\
{\displaystyle d\,\colon M\times M\to \mathbb {R}. }
\\
&
	{d(x,y)=0\iff x=y},\quad \quad \text{ } \text{ } \text{ } \text{ } \text{(identity of indiscernibles)}
  \\
&	 { d(x,y)=d(y,x)}, \quad \quad \quad \quad \quad \quad \text{(symmetry)}\\

&  { d(x,z)\leq d(x,y)+d(y,z)}.  \quad \quad \text{ } \text{(triangle inequality)}
\\
& d(x, y) \geq 0.
\\
x_n & = \frac{1}{n} \rightarrow (1, 1/2, 1/3, 1/4, 1/5, 1/6, ...). \\
d(x_m - x_n) < \epsilon. \\
&x_n = \frac{7 x_{n-1}}{8} + \frac{1}{x_{n-1}}. \\
& x = \frac{7 x}{8} + \frac{1}{x} \\
& \frac{x}{8} = \frac{1}{x}\\
& x^2 = 8.\\
& \phi(a+b) = \phi(a) + \phi(b) \text{ and } \phi(ab) = \phi(a)\phi(b).
\\
& ⟨V, 𝔽, +, ×, - , \textbf{0}   ⟩.\\
\text{Associative law: }  & (u + v) + w = u + (v + w), \\
\text{Commutative law: } & u + v = v + u, \\
\text{Inverse law: } & u + (-u) = \textbf{0}, \\
\text{Identity laws: } & \textbf{0} + u = u, \\
& \textbf{1}  u = u, \\
\text{Distributive laws: } & a (b  u) = (ab) u, \\
& (a + b) u = a  u  + b  u. \\
( 1 - \frac{1}{n} )_{n=1}^\infty. \\
& ‖ \cdot ‖_m \text{, like } ‖ \cdot ‖_2. \\
d(x, y) = 0 \iff x ≡ y. \\
⟨ x, y ⟩. \\
\\
‖ x ‖ = \sqrt{⟨ x, x ⟩}. \\
\text{Associative: } & ⟨au, v ⟩= a⟨u,  v⟩, \text{ where } a \in \mathbb{F},\\
\text{Commutative: } & ⟨u, v⟩ = ⟨v, u⟩, \\
\text{Distributive: } & ⟨u, v + w⟩ = ⟨u, v⟩ + ⟨u, w⟩. \\
⟨u, v⟩ = \sum_{i=1}^n u_i v_i. \\
&   ‖ ⟨  x_i⟩^n_{i=1} ‖_p = \Bigg(
  \sum_{i=1}^n | x_i |^p
  \Bigg)^{\frac{1}{p}}. \\
  \\
  &   ‖ ⟨  x_i⟩^n_{i=1} ‖_2 =
     \sqrt{\sum_{i=1}^n  x_i^2
    }.
\\
‖ f ‖_{sup} = \sup_{x \in X} |f(x)|.    
\\
& L_{p} = \{(f: \mathbb{R}^n \rightarrow \mathbb{R}) : \int_{- \infty}^\infty |f^p(x)|dx < \infty\}. \\
\\
& ‖f‖_{L_p} = \Bigg( \int_{- \infty}^\infty | f^p(x)|dx  \Bigg)^{\frac{1}{p}} \\
\end{align}
$$

$$
\begin{align}
section 16 \\

& dt = \frac{ds}{v(x, y)}. \\
\\
& ds^2 = dx^2 + dy^2 \Rightarrow  ds = \sqrt{1+ \Big(\frac{dy}{dx}\Big)^2} dx,\\
& dt = \frac{\sqrt{1+ \Big(\frac{dy}{dx}\Big)^2} }{v(x, y)} dx. \\
\\

& T = \int_{x_1}^{x_2} \frac{\sqrt{1+ \Big(\frac{dy}{dx}\Big)^2}}{v(x, y)} dx.
\\
& I = \int_{x_1}^{x_2} F(x, y, y') dx, \quad \quad  \text{where } y' = \frac{dy}{dx},\\
\\

& η(x) \text{ where }  η(x_1) = η(x_2) = 0. \\
& \bar{y}(x) = y(x) + \epsilon \eta(x) .\\
& \bar{y} \quad \text{ s.t. } \quad I = \int_{x_1}^{x_2} F(x, \bar y, \bar y') dx \quad \text{ is stationary.}\\
& \frac{dI}{d \epsilon} =0 \rightarrow \frac{dI}{d \epsilon} \Bigr|_{\epsilon=0} =0. \\
& \frac{d  }{d \epsilon} \Bigr|_{\epsilon=0} \int_{x_1}^{x_2} F(x, \bar y, \bar y') dx =0.  \\
\\
\frac{dI}{d \epsilon} \Bigr|_{\epsilon=0} =0 & \Rightarrow
\int_{x_1}^{x_2} \frac{d  }{d \epsilon}  F(x, \bar y, \bar y') \Bigr|_{\epsilon=0}  dx =0 \\
& \Rightarrow
\int^{x_2}_{x_1} \Big[
\frac{\partial F}{\partial \bar y}
\frac{\partial \bar y}{\partial \epsilon} +
\frac{\partial F}{\partial \bar y'}
\frac{\partial \bar y'}{\partial \epsilon}
 \Big] \Bigr|_{\epsilon=0}   dx  = 0 \\
 & \Rightarrow \int^{x_2}_{x_1} \Big[
 \frac{\partial F}{\partial \bar y} \eta +
 \frac{\partial F}{\partial \bar y'} \eta'
 \Big] \Bigr|_{\epsilon=0}   dx  = 0. \\
\\
\\
\int^{x_2}_{x_1}
\frac{\partial F}{\partial \bar y'} \eta' \Bigr|_{\epsilon=0} dx & =
 \frac{\partial F}{\partial \bar y'} \eta \Bigr|^{x_2}_{x_1}
   -
 \int^{x_2}_{x_1} \eta \frac{d}{dx} \frac{\partial F}{\partial \bar y'} \Bigr|_{\epsilon=0} dx = -\int^{x_2}_{x_1} \eta \frac{d}{dx} \frac{\partial F}{\partial \bar y'} \Bigr|_{\epsilon=0}dx.

\\
\\
\frac{dI}{d \epsilon} \Bigr|_{\epsilon=0} =0 & \Rightarrow
\int^{x_2}_{x_1} \eta \Big[
\frac{\partial F}{\partial \bar y'} -
\frac{d}{dx} \frac{\partial F}{\partial \bar y'}

\Big] \Bigr|_{\epsilon=0}  dx = 0 \\
& \Rightarrow
\int^{x_2}_{x_1} \eta \Big[
\frac{\partial F}{\partial y'} -
\frac{d}{dx} \frac{\partial F}{\partial y'}

\Big]   dx = 0, \text{ as } \epsilon = 0, \text{ we can replace $\bar y$ with $y$}
\\
& \Rightarrow
\int^{x_2}_{x_1} \eta \Big[
\frac{\partial F}{\partial y'} -
\frac{d}{dx} \frac{\partial F}{\partial y'}

\Big]   dx = 0.

\\
& \text{Euler-Lagrange Equation: } \quad \quad \frac{\partial F}{\partial y'} -
\frac{d}{dx} \frac{\partial F}{\partial y'}  = 0. \\
F & = \sqrt{1 + (y')^2} \\
L & = \int_{x_1}^{x_2} \sqrt{1 + (y')^2} dx. \\

\frac{\partial F}{\partial y'} -
\frac{d}{dx} \frac{\partial F}{\partial y'} = 0  &
\Rightarrow 0 - \frac{d}{dx} \frac{\partial F}{\partial y'} = 0 \text{ as $F$ does not have any term in $y'$.}  \\
& \Rightarrow - \frac{d}{dx} \Big(
\frac{y'}{\sqrt{1 + (y')^2}}
\Big) = 0\\
& \Rightarrow \frac{y'}{\sqrt{1 + (y')^2}} = c, \quad \text{ where $c$ is a constant.} \\
& \Rightarrow \frac{(y')^2}{1 + (y')^2} = c^2 \\
& \Rightarrow y' = \frac{c}{\sqrt{1-c^2}} \\
& \Rightarrow y = \frac{c}{\sqrt{1-c^2}}x + d, \text{ where $d$ is a constant.} \\
\\
& \frac{d }{d \epsilon } \bar y (x) = \eta(x) \\
& \frac{d }{d \epsilon } \bar y' (x) = \eta'(x).
\\
& d(x, y) = |x - y|, \quad  d(\boldsymbol x, \boldsymbol y) = ‖\boldsymbol x - \boldsymbol y‖= \sqrt{\sum_{i=1}^n (x_i -y_i)^2}, \quad \\
& d(\boldsymbol x, \boldsymbol y) = \max(|x_1 -y_1|, \cdots, |x_n -y_n|). \\
& B_\epsilon(x) = \{y \in X | d(x, y) < \epsilon\}. \\
& \text{For each } x ∈ A, \text{ there is an open ball with } B_\epsilon(x) ⊆ A. \\
& \text{For all $\epsilon > 0$}, x \in X \text{ is called a boundary point for } A \text{ if } \\

& \quad \quad B_\epsilon(x) ∩ A ≠ Ø \text{, and} \\
& \quad \quad B_\epsilon(x) ∩ A^c ≠ Ø, \\
& \delta A = \{x \in X | x \text{ is a boundary point for } A.\}. \\

& A ∪ \delta A  = A. \\
& A ⊆ X \text{ is closed if  } A^c = X \backslash A \text{ is open. } \\
& \bar A = A ∪ \delta A .
\\
& X = (-6, -1) ∪ (0, 3], \\
& A = (0, 3],  \\
& C = (0, 2].  \\
& \delta A =  Ø, \\
& A ∩ \delta A = Ø, \quad A \text{ is a open set.} \\
& A ∪ \delta A = A, \quad A \text{ is an closed set.} \\

& \bar A = A \Rightarrow A \text{ is a closed set.} \\
& \delta C = \{2\}, \quad \bar C = C ∪ \delta C = (0, 2], \\
& \bar C =C  \quad   \Rightarrow \quad  C \text{ is a closed set.}\\
&  (x_1, x_2, x_3, x_4, \cdots ) \text{ or } (x_n)_{n \in \mathbb{N}}, \quad \text{where } x_i \in X. \\
\\

& \text{A sequence $(x_n)_{n \in \mathbb{N}}$ in a metric space $(X, d)$ is called convergent} \\
& \text{if we can find $\tilde x \in X$ and $N \in \mathbb{N}$ such that}  \\
& \quad \quad d(x_n, \tilde{x}) < \epsilon , \\
&\text{ for } \forall \epsilon >0 \text{ and } \forall n ≥ N.\\
\\
& \lim_{n \rightarrow \infty} x_n = \tilde x.
\\
& \quad \quad \quad \quad 10^{1-m}. \\
\end{align}
$$

$$

\begin{align}
Section 16 b \\

& ‖ x ‖ ≥ 0, &\text{(non-negative)}. \\
& ‖ x ‖ = 0 \iff x = 0, & \text{(positive definite)}. \\
& ‖ \lambda \cdot x ‖ = |\lambda| ‖ x ‖, \quad \forall \lambda \in 𝔽, x \in X,  & \text{(absolutely homogeneous).} \\
& ‖ x + y  ‖ ≤ ‖ x  ‖ + ‖ y  ‖, \forall x, y \in X, & \text{(triangle inequality)}. \\
\\
& 𝔽 \in \{ \mathbb{R}, \mathbb{C} \}, x \in X \text{ (a 𝔽-vector space).}\\
& \text{(A vector space with the scalar in the scalar multiplication be $ \mathbb{R}$ or $\mathbb{C}$.)} \\
‖ Ø ‖ & = ‖ x + (-x) ‖ ≤ ‖ x ‖ + ‖ -x ‖ = 2 ‖ x ‖, \\
‖ x ‖ & ≥ 0. \\
& d(x, y) = ‖ x -y  ‖. \\
&d(x_m - x_n) < \epsilon, \quad \quad \epsilon >0 , \exists N \in \mathbb{N} \text{ and } n,m > N. \\

& l^p = \Big\{ (x_i)_{i=1}^{\infty} \Big\}, \quad s.t. \sum_{i=0}^{\infty} |x_i|^p < \infty. \\
‖ x ‖_p & =
 \Big( \sum_{i=1}^{\infty} |x_i|^p \Big)^{\frac{1}{p}}.\\
& (l^p, ‖ \cdot ‖_p).\\
& l^2 = \Big\{ (x_i)_{i=1}^{\infty} \Big\}, \quad s.t. \sum_{i=1}^{\infty} |x_i|^2 < \infty, \quad ‖ x ‖_2 =
 \Big( \sum_{i=1}^{\infty} |x_i|^2 \Big)^{\frac{1}{2}}.
\\
& L^p = \Big\{ f: \mathbb{R}^n \rightarrow \mathbb{R} \Big\}, \quad s.t. \int_{- \infty}^{\infty} |f^p(x)| dx < \infty. \\
& ‖ f ‖_p = \Bigg(
  \int_{- \infty}^{\infty} |f^p(x)| dx
  \Bigg)^{\frac{1}{p}}. \\
  \cos(\alpha) = \frac{ ⟨ x, y⟩}{‖x‖‖y‖}.\\
& ⟨ \cdot, \cdot⟩ : X \times X \rightarrow \mathbb{F}, \quad \mathbb{F} \in \{ \mathbb{R}, \mathbb{C}\} . \\
\\
\text{Positive definiteness: } & ⟨ x, x⟩ ≥ 0. \quad  ⟨ x, x⟩ = 0 \text{ iff } x = 0, \\
\text{Symmetry: } &⟨ x, y⟩ = {⟨ y, x⟩} \text{ for } \mathbb{F} = \mathbb{R} \text{ , or}
\\
& ⟨ x, y⟩ = \overline{⟨ y, x⟩}, \text{ for } \mathbb{F} = \mathbb{C}. \\
\text{Associative: } & ⟨ax, y ⟩= a⟨x,  y⟩, \text{ where } a \in \mathbb{F},\\
\text{Distributive: } & ⟨x, y + z⟩ = ⟨x, y⟩ + ⟨x, z⟩. \\
\\
⟨ x, y⟩ = \sum^n_{i=1} x_i y_i. \\
  \\
  (X, ⟨\cdot, \cdot⟩).
\\
& \Bigg(\mathbb{R}^n, ⟨x, y⟩ = \sum_{i=1}^n x_i y_i \Bigg). \\  
& |⟨x, y⟩| ≤ ‖x‖‖y‖. \\
& ⟨x, y⟩ = 0. \\
& ‖x - y‖ = \sqrt{⟨x-y, x-y⟩}. \\
& ⟨ M, d⟩. \\
X &\in \mathbb{R},\\
A & = (3, 6).  \\
& C[a, b] = \{u: [a, b] \mapsto \mathbb{R} \text{ is continous}\}.\\
 \text{ with the metrix} \\
& d(u, v) = \sup_{x \in [a. b]} |u(x) - v(x)|.
\\
& A \text{ is closed } \Leftrightarrow X \backslash
A \text{ is open, } \\
& A \text{ is open } \Leftrightarrow X \backslash
A \text{ is closed. } \\
x+ 5, x^2 +x. \\
x+ 5 + x^2 +x & = x^2 + 2x + 5, \\
2 (x + 5) & = 2x + 10.\\

& p_{N}(k) = \begin{pmatrix}
N \\
k
\end{pmatrix} 0.5^{N-k}0.5^k = \begin{pmatrix}
N \\
k
\end{pmatrix} 0.5^N.\\
\text{p-value} & = p_{10}(10) + p_{10}(9) + p_{10}(8) + p_{10}(0) + p_{10}(1) + p_{10}(2)\\ &= 2 \times \Big( p_{10}(10) + p_{10}(9) + p_{10}(8)  \Big). \\
\

\end{align}
$$

$$
\begin{align}
Section 17 \\
⟨f, g⟩ = \int_X f(x)g(x) dx, \\

⟨x, y⟩ = \sum_{i=1}^{\infty} x_iy_i \\

& ⟨x, x⟩ = \sum_{i=1}^{\infty} x_ix_i  = \sum_{i=1}^{\infty} |x_i|^2 ≥ 0 \\
& \text{For } |x_i|^2 = 0 \Rightarrow x_i = 0.  \\
& ⟨x, y+z⟩ = \sum_{i=1}^{\infty} x_i(y_i + z_i) = ⟨x, y⟩ + ⟨x, z⟩. \\
& ⟨x, \lambda y⟩ = \sum_{i=1}^{\infty} x_i(\lambda y_i) =  \lambda ⟨x, y⟩. \\
& f(B_δ(a))⊆ Bε(f(a)). \\
& (x_n)_{n=1}^{\infty} ,\\
& \text{As } x_n  \rightarrow \tilde x, \quad f(x_n) \rightarrow f(\tilde x)
\text{ as n } \rightarrow \infty.
\\
& ‖ f ‖_{sup} = \sup_{x \in X} | f(x) |. \\
& x_k = 0.r_1r_2 \cdots r_k. \\
& \lim_{k \rightarrow \infty} x_k = x, \text{ where } x_k \in \mathbb{Q}, x \in \mathbb{R}. \\
& \overline{Y} = X. \\
& \text{Given } (X, ‖ · ‖_X) \text{ and } (Y, ‖ · ‖_Y) \text{ be two normed spaces,}\\
& \text{A linear operator } T: X \rightarrow Y \text{ is called bounded if there exists a constant } C \\
&\text{such that } \\
\\
& ‖Ax‖_Y ≤ C‖x‖_X \text{ for } \forall x ∈ X. \\
& T: X \rightarrow Y \text{ (linear operator)}, \\
& ‖T‖ = \sup \Bigg\{ \frac{‖Tx‖_Y}{‖x‖_X} \mid x \in X, x ≠ 0 \Bigg\}. \\
\\
& ‖T‖ = \sup_{‖x‖_X = 1} \Bigg\{ ‖Tx‖_Y \mid x \in X, x ≠ 0 \Bigg\}. \\
& \forall x, z \in X, \lambda \in \mathbb{F}, \\
& T (x + z)  = Tx + Tz, \\
& T (\lambda x) = \lambda Tx. \\  
\\
\text{Absolute-value norm : }& ‖x‖ = |x| \text{ for one-dimensional vector spaces,}  
\\
\text{Manhattan norm: }& ‖x‖_1 = \sqrt{ \sum_{i=1}^n | x_i| },\\
\text{Euclidean norm ($l^2$ norm): }& ‖x‖_2 = \sqrt{ \sum_{i=1}^n x_i^2 },\\
\text{p-norm: }& ‖x‖_p = \Bigg({\sum_{i=1}^n |x_i^p| }\Bigg)^{\frac{1}{p}},\\
\text{Max norm ($l_{\infty}$ norm): }& ‖x‖_\infty = \max(|x_1|, |x_2|, \cdots, |x_n|),\\

\text{Frobenius norm): }& ‖x‖_F =  \Bigg({\sum_{ij} A_{ij}^2 }\Bigg)^{\frac{1}{2}}.\\
\\
\\
& ‖x‖_{\infty} = \sup(|x_1|, |x_2|, |x_3|, ...).
\\
\\
T& : X \rightarrow Y, \\
X &= \Big(C([0, 1], \mathbb{F}), ‖ \cdot ‖_\infty \Big), Y = \Big(\mathbb{F}, |\cdot| \Big), \mathbb{F} \in \{\mathbb{R}, \mathbb{C} \},  \\
& \text{For } g \in X \text{ with } g(t) ≠ 0, \quad \forall t \in [0, 1], \text{ define } \\
& T_g: X \rightarrow Y \text{ as } T_g(f) = \int^1_0 g(t) \cdot f(t) dt. \\

‖ T_g ‖ & = \sup \Bigg\{
  \frac{|T_g(f)| }{ ‖ f ‖_\infty } \bigg\rvert f \in X, f \neq 0 \Bigg\}\\

   & = \sup \Bigg\{
    |T_g(f)  \bigg\rvert f \in X, f \neq 0 , { ‖ f ‖_\infty = 1}
\Bigg\}, \text{ for simplicity}\\

 & = \sup \Bigg\{
 \bigg\rvert \int^1_0 g(t) \cdot f(t) dt \bigg\lvert \quad
    \bigg\rvert f \in X, f \neq 0 , { ‖ f ‖_\infty = 1}
\Bigg\} \\
& \leq  \int^1_0 |g(t)| dt \quad
   \bigg\rvert f \in X, f \neq 0 , { ‖ f ‖_\infty = 1}
\Bigg\} \quad \text{ as } ‖ f ‖_\infty = 1.\\
h(t) &= \frac{g(t)}{| g(t)|} \Rightarrow  ‖ h ‖_\infty = 1. \\
 ‖ T_g ‖   \ge | T_g(h)| & = \Bigg\lvert
\int^1_0 g(t) h(t) dt \Bigg\rvert
= \int^1_0  g(t)  \frac{g(t)}{|g(t)|} dt \\
& = \int^1_0 \frac{| g(t) |^2}{|g(t)|} dt \\
‖ T_g ‖  &\ge \int^1_0 | g(t) | dt.
\\
& ‖ T_g ‖ = \int^1_0 | g(t) | dt.
\\
\\
& T f(x) = \int_a^x f(y) dy, \quad f \in C[a, b]. \\
& u(x, y): [0, 1] \times [0, 1]  \rightarrow \mathbb{R} \\
& T f(x) = \int_a^x K(x, y) f(y) dy, \quad f \in C[a, b]. \\
(1 - \frac{1}{n})_{n=1}^\infty, \\
&‖ A ‖_F = \Big( \sum_{i=1}^m \sum_{j=1}^n |A_{ij}|^2 \Big)^\frac{1}{2}. \\
‖ Ax ‖^2_2 & = \sum_{i=1}^m \Bigg{|} \sum_{j=1}^n A_{ij}x_j \Bigg{|}^2 \\
& \leq  \sum_{i=1}^m \Big(
  \sum_{j=1}^n
  |A_{ij}|^2 ‖ x ‖^2_2\Big), \quad \text{ (by Cauchy-Schwarz inequality)} \\
‖ Ax ‖_2  & \leq  ‖ A ‖_F ‖ x ‖_2.\\

‖ T ‖ &= \sup \Bigg(\frac{‖ Ax ‖_2 }{‖ x ‖_2} \Bigg) \leq  ‖ A ‖_F. \\
‖ Tu - Tv ‖_Y & = ‖ T(u - v) ‖_Y, \quad \text{ (by linearity)} \\
& \leq C ‖ (u - v) ‖_X. \\

& \begin{pmatrix}
x_1  \\
x_2
\end{pmatrix} \rightarrow
\begin{pmatrix}
x_1  \\
x_2  \\
x_1 x_2  \\
x_1^2  \\
x_2^2  \\
x_1 x_2^2  \\
x_1^2 x_2  \\
x_1^3  \\
x_2^3  
\end{pmatrix}. \\

& \begin{pmatrix}
\theta_1  &
\theta_2  &
\theta_3  &
\theta_4  &
\theta_5  &
\theta_6  &
\theta_7  &
\theta_8  &
\theta_9  
\end{pmatrix}
\begin{pmatrix}
x_1  \\
x_2  \\
x_1 x_2  \\
x_1^2  \\
x_2^2  \\
x_1 x_2^2  \\
x_1^2 x_2  \\
x_1^3  \\
x_2^3  
\end{pmatrix} ≥ 0 ? \\

&⟨ \theta, 𝜙(x)  ⟩ \quad \text{ where }
𝜙(x_1, x_2) = (x_1,
x_2,
x_1 x_2,
x_1^2,
x_2^2,
x_1 x_2^2,
x_1^2 x_2,
x_1^3,
x_2^3)^T.
\\
\Bigg\langle
\begin{pmatrix}
1  \\
x_1  \\
x_2  \\
\end{pmatrix},
\begin{pmatrix}
1  \\
y_1  \\
y_2  \\
\end{pmatrix}
\Bigg\rangle = \Bigg\langle
\begin{pmatrix}
1  \\
1  \\
2  \\
\end{pmatrix},
\begin{pmatrix}
1  \\
3  \\
4  \\
\end{pmatrix}
\Bigg\rangle = 12.
\\
\\
⟨x, y⟩ \rightarrow ⟨\phi(x), \phi(y)⟩.
\\
\\
\\
\Bigg\langle \phi(x), \phi(y) \Bigg\rangle =
\Bigg\langle
\begin{pmatrix}
1 \\
\sqrt{2} \times 1 = \sqrt{2}   \\
\sqrt{2} \times 2 = 2 \sqrt{2} \\
\sqrt{2} \times 1 \times 2 =  2 \sqrt{2} \\
1^2 = 1 \\
2^2 = 4
\end{pmatrix},
\begin{pmatrix}
1 \\
\sqrt{2} \times 3 = 3 \sqrt{2}  \\
\sqrt{2} \times 4 = 4 \sqrt{2}  \\
\sqrt{2} \times 3 \times 4 = 12 \sqrt{2}  \\
3^2 = 9 \\
4^2 = 16
\end{pmatrix} \Bigg\rangle
 = 1 + 6 + 16 + 48 + 9 + 64 = 144. \\
\\

\phi \Bigg(
\begin{pmatrix}
1 \\
x_1  \\
x_2
\end{pmatrix}
\Bigg) =
\begin{pmatrix}
1 \\
\sqrt{2} x_1  \\
\sqrt{2} x_2  \\
\sqrt{2} x_1 x_2 \\
x_1^2  \\
x_2^2  \\
\end{pmatrix}. \\
\\

(x_1, x_2) \rightarrow (x_1^2, \sqrt{2} x_1 x_2, x_1)
\\
\\

\\
k(x, y) = \Bigg\langle
\begin{pmatrix}
1 \\
x_1  \\
x_2
\end{pmatrix}
,
\begin{pmatrix}
1 \\
y_1  \\
y_2
\end{pmatrix}\Bigg\rangle^2 =  ⟨\phi(x), \phi(y)⟩.
\\
\\
k(x, y) =  ⟨\phi(x), \phi(y)⟩_H. \\

⟨x, y⟩ = 0 \text{ if } x ⊥ y \quad \text{ (orthogonal)}.\\
\text{For } U, V ⊆ X, U ⊥ V \text{ if } x ⊥ y \text{ for } ∀ x ∈ U, ∀ y ∈ V. \\
U^⊥ = \{ x \in X | ⟨x, u⟩=0 \text{ for } ∀ u \in U  \}.
\\

\\

\end{align}
$$

$$
Section 18 \\
\\
\Bigg\langle
\begin{pmatrix}
1 \\
x_1  \\
x_2
\end{pmatrix},
\begin{pmatrix}
1 \\
y_1  \\
y_2
\end{pmatrix}
\Bigg\rangle^2 =
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
\\
\\
\begin{align}
x, y, z \in H &\\
\\
\text{Linearity: }
& \quad ⟨
a x + by, z
⟩_H = a ⟨x, z ⟩_h + b ⟨y, z ⟩_H   ,\\
\text{Symmetry:} & \quad ⟨x, y⟩_H = ⟨y, x⟩_H  , \\
\text{Positive definite:} & \quad ⟨x, x⟩_H ≥ 0, \quad ⟨x, x⟩_H = 0 \text{ iff } x =0  . \\
\\
\text{Polynomials of degree exactly $d$: } \quad & k(x, y) = ⟨x, y ⟩^d, \\
 \text{Polynomials of degree up to $d$: } \quad & k(x, y) = \Big(⟨x, y ⟩ +1 \Big)^d. \\
\text{Gaussian kernels: } \quad &  k(x, y) =  \exp \Bigg(
  - \frac{‖x - y ‖^2_2 }{2 \sigma^2}
  \Bigg).
\\

&k(x, y) = \int_x \int_y k(x, y) g(x) g(y) dx dy ≥ 0, \quad \text{for $ \forall g \in $ \{nonzero continuous functions\}.} \\
& \begin{pmatrix}
k(x_1, x_1) & k(x_1, x_2) & \cdots & k(x_1, x_n)  \\
k(x_2, x_1) & k(x_2, x_2) & \cdots & k(x_2, x_n)  \\
 &  \ddots &  &  \\
k(x_n, x_1) & k(x_n, x_2) & \cdots & k(x_n, x_n)  \\
\end{pmatrix} \text{ is positive semidefinite for $\forall x_1, x_2, \cdots , x_n \in X$}.
\\
\mathcal{L}(\boldsymbol \theta, \boldsymbol X, \boldsymbol y) & = (\boldsymbol y - \boldsymbol X \boldsymbol \theta)^\top (\boldsymbol y - \boldsymbol X \boldsymbol \theta), \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol \theta}
 & = -2 \boldsymbol X^\top \boldsymbol y + 2\boldsymbol X^\top \boldsymbol X \boldsymbol \theta = 0.
 \\
 \\
 \boldsymbol X^\top \boldsymbol X \boldsymbol \theta & = \boldsymbol X^\top \boldsymbol y. \\
\\
 \boldsymbol \theta & = (\boldsymbol X^\top \boldsymbol X)^{-1} \boldsymbol X^\top \boldsymbol y, \quad \text{ given }  \boldsymbol X^\top \boldsymbol X \text{ is invertible.}
 \\
&= \boldsymbol X^\top \boldsymbol X (\boldsymbol X^\top \boldsymbol X)^{-2} \boldsymbol X^\top \boldsymbol y
&= \boldsymbol X^\top \boldsymbol \alpha \\
&= \boldsymbol X^\top \boldsymbol \alpha, \quad \text{ where } \boldsymbol \alpha = \boldsymbol X (\boldsymbol X^\top \boldsymbol X)^{-2} \boldsymbol X^\top \boldsymbol y\\
&= \sum_{i=1}^n \alpha_i \boldsymbol {x_i}. \\
\\
f(\boldsymbol x)
&= ⟨\boldsymbol \theta, \boldsymbol x⟩ \\
&= ⟨\sum_{i=1}^n \alpha_i \boldsymbol {x_i}, \boldsymbol x⟩ \\
&= \sum_{i=1}^n \alpha_i ⟨ \boldsymbol {x_i}, \boldsymbol x⟩ \\
& = \Big( \boldsymbol X (\boldsymbol X^\top \boldsymbol X)^{-2} \boldsymbol X^\top \boldsymbol y \Big)^\top ⟨ \boldsymbol {x_i}, \boldsymbol x⟩\\
&= \Big(\boldsymbol y^\top \boldsymbol X (\boldsymbol X^\top \boldsymbol X)^{-2} \boldsymbol X^\top  \Big) \boldsymbol K,    \quad (X^\top X \text{ is symmetry)} \\
& \text{ where } \boldsymbol K_i =  ⟨ \boldsymbol {x_i}, \boldsymbol x⟩. \\
 \min_\theta \mathcal{L}_{\lambda} (\theta, D) & =
\min_\theta \lambda ‖ \theta ‖^2 + \sum_{i=1}^n (y_i - f(x_i))^2 \\

& =
\min_\theta \lambda ‖ \theta ‖^2 + (\boldsymbol y − \boldsymbol {X\theta})^\top (\boldsymbol y − \boldsymbol {X\theta}). \\

\frac{ \partial \mathcal{L} }{\partial \boldsymbol {\theta}  } & = 2  \lambda \boldsymbol {\theta} - 2  \boldsymbol X^\top \boldsymbol y + 2 \boldsymbol {X^\top}
\boldsymbol {X} {\theta} = 0
\\
& \boldsymbol {X^\top X \theta} -\boldsymbol X^\top \boldsymbol  y + \lambda \boldsymbol {\theta}  = 0 \\
 & (\boldsymbol{X^\top X} + \lambda \boldsymbol I_n) \boldsymbol \theta = \boldsymbol X^\top \boldsymbol  y \\
\\
\boldsymbol \theta & = (\boldsymbol{X^\top X} + \lambda \boldsymbol I_n)^{-1} \boldsymbol X^\top \boldsymbol y. \\
f(\boldsymbol x) & = ⟨\boldsymbol \theta, \boldsymbol x⟩ =
 \boldsymbol \theta^T  \boldsymbol x =
  \boldsymbol y^\top  \boldsymbol X ( \boldsymbol X^\top  \boldsymbol X + \lambda  \boldsymbol I_n)^{-1}  \boldsymbol x.
\\
\boldsymbol {X^\top X \theta} + \lambda \boldsymbol {\theta} &  = \boldsymbol X^\top \boldsymbol  y \\
 \boldsymbol {\theta} &  = \lambda^{-1} \boldsymbol X^\top (\boldsymbol y - \boldsymbol{X\theta}) = \boldsymbol{X}^\top \boldsymbol{\alpha}. \\
& \text{where } \\
\boldsymbol{\alpha} &  = \lambda^{-1} (\boldsymbol y - \boldsymbol{X\theta}) \\
\lambda \boldsymbol{\alpha} & =  (\boldsymbol y - \boldsymbol{X X^{\top}\alpha}) \quad \text{ where } \boldsymbol \theta = \boldsymbol X^\top \boldsymbol \alpha
\\
(\boldsymbol{X X^\top} + \lambda \boldsymbol I_n)
\boldsymbol{\alpha} & = \boldsymbol y \\
\boldsymbol{\alpha} &= (\boldsymbol{G} + \lambda \boldsymbol I_n )^{-1} \boldsymbol{y} \quad \text{ where }
\boldsymbol G = \boldsymbol{X X^\top}. \\
\\
f(\boldsymbol x) & = (\boldsymbol \theta, \boldsymbol x) = \Bigg\langle \sum^n_{i=1} \alpha_i  \boldsymbol x_i, \boldsymbol x \Bigg\rangle = \sum^n_{i=1} \alpha_i \langle \boldsymbol x_i, \boldsymbol x \rangle = \boldsymbol y^\top \Big(
\boldsymbol G + \lambda \boldsymbol I_l  
\Big)^{-1} \boldsymbol k
\\ & \text{where } k_i = \langle \boldsymbol x_i, \boldsymbol x  \rangle.
\\
& \langle \boldsymbol x, \boldsymbol y \rangle \rightarrow \langle \phi(\boldsymbol x), \phi(\boldsymbol y) \rangle, \quad \quad \boldsymbol G_{ij} =  \langle \phi(\boldsymbol x_i), \phi(\boldsymbol x_j) \rangle.
\\
f_{\boldsymbol z} (\boldsymbol x) = \langle  \boldsymbol x , \boldsymbol z \rangle.
\\
\\
\boldsymbol K = & \begin{pmatrix}
k(\boldsymbol x_1, \boldsymbol x_1) & k(\boldsymbol x_1, \boldsymbol x_2) & \cdots & k(\boldsymbol x_1, \boldsymbol x_n)  \\
k(\boldsymbol x_2, \boldsymbol x_1) & k(\boldsymbol x_2, \boldsymbol x_2) & \cdots & k(\boldsymbol x_2, \boldsymbol x_n)  \\
 &  \ddots &  &  \\
k(\boldsymbol x_n, \boldsymbol x_1) & k(\boldsymbol x_n, \boldsymbol x_2) & \cdots & k(\boldsymbol x_n, \boldsymbol x_n)  \\
\end{pmatrix}, \\
\\
& K_{ij} =  \langle \phi(\boldsymbol x_i), \phi(\boldsymbol x_j) \rangle = k(\boldsymbol x_i, \boldsymbol x_j).
\\
G_{ij} = \langle \boldsymbol x_i, \boldsymbol x_i \rangle.
\\
\boldsymbol v^\top \boldsymbol G \boldsymbol v &=
\sum^n_{i, j = 1} v_i v_j \boldsymbol G_{ij} \\
&= \sum^n_{i, j = 1} v_i v_j \langle \phi(\boldsymbol x_i), \phi(\boldsymbol x_j) \rangle \\
&= \Bigg\langle
\sum^n_{i= 1} \boldsymbol v_i \phi(\boldsymbol x_i),
\sum^n_{j= 1} \boldsymbol v_j \phi(\boldsymbol x_j)\Bigg\rangle \\
& = \Bigg\| \sum^n_{i= 1} v_i \phi(\boldsymbol x_i) \Bigg\|^2 ≥ 0. \\


\\
\\
\end{align}

$$

$$

\begin{align}

section 19 \\
\\
\boldsymbol{v}^\top \boldsymbol{A  v}  =
\boldsymbol {v^{\top} B^{\top} B v} = \| \boldsymbol{Bv} \|^2 ≥ 0.\\

\langle \boldsymbol x, \boldsymbol x \rangle = 0 \text{ iff } \boldsymbol x = 0. \\
K: X \times X \rightarrow \mathbb{R}. \\
\\
f(\boldsymbol x)  = \sum^n_{i=1} \alpha_i \langle \boldsymbol x_i, \boldsymbol x \rangle, \quad \text{ where } \alpha_i \in \mathbb{R}. \\
f(x) = f_1 x_1 + f_2 x_2 + f_3 x_1 x_2. \\
f(\cdot) = [f_1, f_2, f_3]^\top. \\
f(x) = f(\cdot)^\top \phi (x) = \langle (\cdot) \rangle
f(x) = f(\cdot)^\top \phi (x) = \langle f(\cdot), \phi (x) \rangle \\
\boldsymbol \theta = \boldsymbol X^\top \boldsymbol \alpha, \quad \text{ where } \boldsymbol \alpha \in \mathbb{R}^n. \\
C_{\boldsymbol z}(\boldsymbol x) = \langle \boldsymbol x, \boldsymbol z \rangle . \\
f: X \rightarrow \mathbb{R} \\

& \text{$\mathcal{H}$ be a vector space over ℝ and $f, g, h ∈\mathcal{H}$.} \\
& ⟨·, ·⟩ : \mathcal{H} × \mathcal{H} → \mathbb{R}, \\
& ⟨ a f + b g, h⟩_{\mathcal{H}} = a ⟨ f, h⟩_{\mathcal{H}} + b ⟨ g, h ⟩_{\mathcal{H}} & \text{(linear)}, \\
& ⟨ f, g⟩_{\mathcal{H}} = ⟨ g, f⟩_{\mathcal{H}} & \text{(symmetry)}, \\
& ⟨ f, f⟩_{\mathcal{H}} \ge 0, \quad ⟨ f, f⟩_{\mathcal{H}} =0 \text{ iff } f =0, & \text{(positive definiteness)}. \\
& K(\cdot, y) = [y_1, y_2, y_1 y_2]^\top = \phi(y). \\
& \langle K(\cdot, y), \phi(x) \rangle_{\mathcal{H}} = a x_1 + bx_2 + c x_1 x_2, \quad \text{ where } a = y_1, b=y_2, \text{ and } c=y_1y_2. \\
\\
& \langle K(\cdot, x), \phi(y) \rangle_{\mathcal{H}} = u y_1 + v y_2 + w y_1 y_2, \quad \text{ where } u = x_1, v=x_2, \text{ and } w=x_1x_2. \\
& K(x, y) = \langle K(\cdot, x), \phi(y) \rangle_{\mathcal{H}} = \langle K(\cdot, x), K(\cdot, y)\rangle_{\mathcal{H}} =  \langle \phi(x), \phi(y)  \rangle_{\mathcal{H}}. \\
& K(x, y)  =  \langle \phi(x), \phi(y)  \rangle_{\mathcal{H}}. \\
& l: X \rightarrow \mathbb{F}, \quad \mathbb{F} \in \{ \mathbb{R}, \mathbb{C}\}, \text{ there is an exactly one } x_l \in X, \\
\\
& l(x) = \langle x_l, x \rangle \text{ for } \forall x \in X.\\ \text{ and } \| l \|_{X \rightarrow \mathbb{F} } = \| x_l \|_{X}. \\
& d(x, y) = \| x - y \|. \\
& \| f \|_p = \Bigg(

  \int^b_a \mid f(x) \mid^p dx
  \Bigg)^{1/p}. \\
&  \Bigg(\mathbb{R}, \mid \cdot \mid \Bigg), \\
&  \Bigg(\mathbb{R}^n, \| \boldsymbol x \|_p = \Big( \sum_{i=1}^n \mid \boldsymbol x_i \mid^p \Big)^{1/p} \Bigg). \\
& \Bigg(
 C[a, b], \langle x, y \rangle = \int^b_a f(x) g(x) dx
\Bigg) \\
& l^2 = \Big\{ (x_i)_{i=1}^{\infty} \Big\}, \quad \sum_{i=0}^{\infty} |x_i|^2 < \infty, \quad  \langle \{ (x_i)_{i=1}^{\infty} \}, \{(y_i)_{i=1}^{\infty} \} \rangle_{l^2} = \sum_{i=1}^\infty x_i y_i \\

&\text{For } g ∈ X, A_g : X → \mathbb{R}, \text{ with } A_g f \text{ define as  } A_g f  = \langle f, g \rangle_X, \\
& A_g \text{ is a linear functional}. \\
A_g(a f_1 + b f_2) & =  \langle af_1+bf_2, g \rangle_X \\
&=  a \langle f_1, g \rangle_X + b \langle f_2, g \rangle_X \\
&= a A_g f_1 + b A_g f_2. \\
& \mathcal{F}' = \{L: \mathcal{F} \rightarrow \mathbb{R} \text{ or } \mathbb{C} \mid L \text{ is linear and bounded.} \}.  \\

& (\theta_1 \quad \theta_2 \quad \theta_3 \quad \cdots \quad \theta_n)
\begin{pmatrix}
x_1  \\
x_2\\
\vdots \\
x_n \\
\end{pmatrix} = b, \quad b ∈ ℝ.\\
& L(\boldsymbol x) = b. \\
& L: X \rightarrow \mathbb{F}, \quad \mathbb{F} \in \{ \mathbb{R}, \mathbb{C}\}, \text{ there is an exactly one } g \in X, \\
\\
&k(x, y) = \langle \phi(x), \phi(y) \rangle_{H}. \\
\\
& k: X \times X \rightarrow \mathbb{R}.\\
& \boldsymbol Q^\top \boldsymbol Q = \boldsymbol Q^\top \boldsymbol Q = \boldsymbol I. \\
& k: X, X \rightarrow \mathbb{R}. \\
& k(x_i, x_j) = \Big\langle \phi(x_i), \phi(x_j) \Big\rangle_{\mathcal{H}}
\end{align}

$$
