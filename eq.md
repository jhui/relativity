$$
\begin{align}
&\text{With ML, we predict a score } R =r(x)\\
\\
\text{average admiss. rate for women} &= \sum_i \text{dept. } i \text{ app. rate for women} \times  \text{dept. } i \text{ admiss. rate for women}
\\
\text{average admiss. rate for women} &= 0.9 \times 0.1 + 0.1 \times 0.5 = 0.14 \\
\text{average admiss. rate for men} &= 0.2 \times 0.08 + 0.8 \times 0.4 = 0.336 \\
\\
\text{dept. } i \text{ admiss. rate for women} & > \text{dept. } i \text{ admiss. rate for men }
&\text{And deicsion } D=1 \text{ (a loan is approved) when } R > t \\
\\
& \text{Require } D \text{ to be independent of } A \text{ (independence)} \\
\\
& \text{Require } D \text{ to be independent of } A \text{ given } Y  \text{ (separation)  } \\
\\
& \text{Require } Y \text{ to be independent of } A \text{ given } R  \text{ (sufficiency)  } \\
\end{align}
$$

$$
\text{Bayes optimal score } R = E[Y|x] \\
$$
$$
\begin{align}
P(Y=1|R=r) & =r   \quad \text{(calibration)}\\
\\
P(Y=1|R=r) & = r   \quad \text{(calibration)}\\
\\
P(Y=1|R=1, A=a) & = P(Y=1|R=1, A=b)   \quad \text{(Predictive parity)}\\
\\
P(Y=y|R=r, A=a) & =r   \quad \text{(calibration by group)}\\
\\
P(D=1|Y=1, A=a) & = P(D=1|Y=1, A=b)  \quad \text{equal opportunity/equalized odd}\\
\\
P(D=1|Y=0, A=a) & = P(D=1|Y=0, A=b)  \quad \text{equal false positive rate}\\
P(D=0|Y=1, A=a) & = P(D=0|Y=1, A=b)  \quad \text{equal false negative rate}\\
\\
| P(D=1|A=a) - P(D=1|A=b) | & ≤ ε  \\
\\
P(D=1|A=a) & = P(D=1|A=b)  \\
x: & \quad \text{describes a sample.} \\
Y: & \quad \text{is the outcome variable.} \\
  & \quad \text{a.k.a ground truth on whether the loan is repaid.} \\
& \quad \text{We want to predict } Y \text{ from } x.\\
A: & \quad \text{a membership class, say } A \text { belongs to male or female}.
\\
pred.+ &= TP + FP \\
total+ &= TP + FN \\
total- &= TN + FP \\
\\
\text{accuracy} &= \frac{TP+TN}{total} = \frac{75+35}{150} \\
\\
\text{precision} &= \frac{TP}{pred. +} = \frac{75}{90} \\
\\
\text{recall/sensitivity/TPR} &= \frac{TP}{total +} = \frac{75}{100} \\
\\
\text{specificity/TNR} &= \frac{TN}{total -} = \frac{35}{50}
\\
\\
F1 &= 2 \times \frac{precision \times recall}{precision + recall}
\\
\\
FPR &= \frac{FP}{total -} = 1 - \text{TNR} = \frac{15}{50}
\\
\\
\text{miss rate/FNR} &= \frac{FN}{total +} = \frac{25}{100}
\\
\\
\text{prevalence} &= \frac{total +}{total} = \frac{100}{150}
\\
\\
\text{misclassification rate} & = \frac{FP+FN}{total} = \frac{15+25}{150}
\\
\\
FP & = \text{type 1 error} \\
FN & = \text{type 2 error} \\
\\
\\
\text{Speed of light} &= \frac{\text{distance travel}}{\text{time}} \\
F &= ma = \frac{d^2 x}{d t^2} \\
x' & = x - Vt \\
y' & = y \\
z' & = z \\
t' &= t \\
x' & = x + v_xt \\
y' & = y + v_yt \\
z' & = z + v_zt \\
t' &= t \\
∇ \cdot \textbf{E} &=  0 \\
∇ \times \textbf{E} &= - \frac{∂ \textbf{B}}{∂t} \\
∇ \cdot \textbf{B} &= 0 \\
∇ \times \textbf{B} &= μ_0 ε_0 \frac{∂ \textbf{E}}{∂t} \\
\end{align}
$$
