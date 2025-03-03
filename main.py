import streamlit as st
import sympy as sp
from sympy.integrals.manualintegrate import integral_steps
import numpy as np
import matplotlib.pyplot as plt

# Create three tabs: Matrix Calculator, Differential Equations & Dynamical Systems, and Notes.
tab1, tab2, tab3 = st.tabs(["Matrix Calculator", "Differential Equations & Dynamical Systems", "Notes"])

###############################################
# TAB 1: Matrix Calculator (unchanged from before)
###############################################
with tab1:
    st.title("Matrix Calculator: Eigen, Jordan & Matrix Exponential")
    st.markdown(
        """
        Enter your matrix below. Each row should be on a new line with entries separated by commas or spaces.
        For example:
        ``` 
        1, 2
        3, 4
        ```
You can enter fractions (e.g. `1/2`) if you check the "Use fractions" box.
        """
    )

    # Input area for the matrix
    matrix_input = st.text_area("Matrix Input", value="1, 2\n3, 4", height=150)
    use_fractions = st.checkbox("Use fractions (symbolic)", value=True)

    if st.button("Calculate", key="matrix_calc"):
        # Parse the input into a matrix
        rows = matrix_input.strip().splitlines()
        matrix_data = []
        for row in rows:
            if row.strip() == "":
                continue
            parts = row.split(',')
            if len(parts) == 1:
                parts = row.split()
            row_data = []
            for item in parts:
                item = item.strip()
                if use_fractions:
                    try:
                        val = sp.Rational(item)
                    except Exception:
                        val = sp.nsimplify(float(item))
                else:
                    try:
                        val = float(item)
                    except Exception:
                        val = float(item)
                row_data.append(val)
            matrix_data.append(row_data)
        
        # Create a sympy Matrix from the parsed data
        A = sp.Matrix(matrix_data)
        
        st.subheader("Matrix A")
        st.latex(sp.latex(A))
        
        # Eigenanalysis
        eig_data = A.eigenvects()
        st.subheader("Eigenvalues, Multiplicities and Eigenvectors")
        for eigenval, alg_mult, eigenvecs in eig_data:
            geo_mult = len(eigenvecs)
            st.write("**Eigenvalue:**", eigenval)
            st.write("Algebraic multiplicity:", alg_mult)
            st.write("Geometric multiplicity:", geo_mult)
            for vec in eigenvecs:
                st.write("Eigenvector:", vec)
            st.write("---")
        
        # Fundamental matrix (matrix exponential)
        t = sp.symbols('t', real=True)
        M_t = (A * t).exp()
        st.subheader("Fundamental Matrix $M(t)=e^{At}$")
        st.latex(sp.latex(M_t))
        
        # M(0) and its inverse
        M0 = M_t.subs(t, 0)
        st.subheader("M(0) and its Inverse $M(0)^{-1}$")
        st.write("M(0):")
        st.latex(sp.latex(M0))
        try:
            M0_inv = M0.inv()
            st.write("M(0)$^{-1}$:")
            st.latex(sp.latex(M0_inv))
        except Exception as e:
            st.write("M(0) is not invertible.")
        
        # Matrix powers
        A2 = sp.simplify(A**2)
        A3 = sp.simplify(A**3)
        st.subheader("Matrix Powers")
        st.write("A$^2$:")
        st.latex(sp.latex(A2))
        st.write("A$^3$:")
        st.latex(sp.latex(A3))
        
        # Inverse of A
        st.subheader("Inverse of Matrix A")
        try:
            A_inv = A.inv()
            st.latex(sp.latex(A_inv))
        except Exception as e:
            st.write("Matrix A is not invertible.")
        
        # Jordan form & generalized eigenvectors
        st.subheader("Jordan Form & Generalized Eigenvectors")
        try:
            P, J = A.jordan_form()
            st.write("Transformation Matrix P (columns are eigenvectors/generalized eigenvectors):")
            st.latex(sp.latex(P))
            st.write("Jordan Normal Form J:")
            st.latex(sp.latex(J))
            st.markdown(
                """
In the matrix **P**, for each Jordan block corresponding to an eigenvalue:
- The **first column** is a proper eigenvector.
- Subsequent columns are generalized eigenvectors.
                """
            )
            exp_Jt = sp.exp(J * t)
            M_t_jordan = sp.simplify(P * exp_Jt * P.inv())
            st.write("Fundamental Matrix via Jordan decomposition:")
            st.latex(sp.latex(M_t_jordan))
        except Exception as e:
            st.write("Could not compute the Jordan form and generalized eigenvectors.")
        
        # Diagonalization
        st.subheader("Diagonalization of Matrix A")
        try:
            P_diag, D = A.diagonalize()
            st.write("Diagonal Matrix D:")
            st.latex(sp.latex(D))
            st.write("Matrix of Eigenvectors P:")
            st.latex(sp.latex(P_diag))
            st.write("Verification: $ A = P D P^{-1} $")
            st.latex(sp.latex(A - P_diag * D * P_diag.inv()))
        except Exception as e:
            st.write("Matrix A is not diagonalizable.")

###############################################
# TAB 2: Differential Equations & Dynamical Systems (modified)
###############################################
with tab2:
    st.title("Differential Equations, Integrals & Dynamical Systems")
    
    # Updated radio options without Phase Portrait and Nonlinear Simulation
    prob_type = st.radio("Select Problem Type:", 
                         ["Differential Equation", "Integral", "Equilibrium Analysis", "Slope Field"],
                         key="prob_type")
    
    ##########################################
    # Differential Equation Solver
    ##########################################
    if prob_type == "Differential Equation":
        st.markdown("### Differential Equation Solver")
        st.markdown(
            """
Enter a differential equation in terms of **y(t)**.
For more specialized equations, select the type below:
- **General:** Solve using `dsolve`
- **Separable/Linear:** (Assumed to be in standard form)
- **Bernoulli:** For equations of the form `y'(t) = p(t)*y + q(t)*y^n`
- **Riccati (with known particular solution):** Requires you to provide one known particular solution.
            """
        )
        ode_type = st.selectbox("Select ODE Type:", 
                                ["General", "Separable", "Linear", "Bernoulli", "Riccati (with known particular solution)"],
                                key="ode_type")
        de_input = st.text_input("Enter the differential equation (e.g., `y'(t) - y(t) = 0`)", value="y'(t) - y(t) = 0", key="de_input")
        if ode_type == "Bernoulli":
            n_input = st.text_input("Enter the exponent n (for y^n term)", value="2", key="n_input")
        if ode_type == "Riccati (with known particular solution)":
            particular = st.text_input("Enter a known particular solution x1(t):", value="", key="riccati_particular")
        
        if st.button("Solve Differential Equation", key="solve_de_extended"):
            try:
                t = sp.symbols('t')
                y = sp.Function('y')(t)
                # Replace y'(t) with sp.diff(y,t)
                parts = de_input.split('=')
                if len(parts) == 2:
                    lhs = sp.sympify(parts[0].replace("y'(t)", "sp.diff(y,t)"), locals={'y': y, 't': t})
                    rhs = sp.sympify(parts[1], locals={'y': y, 't': t})
                    eq = sp.Eq(lhs, rhs)
                else:
                    eq = sp.sympify(de_input.replace("y'(t)", "sp.diff(y,t)"), locals={'y': y, 't': t})
                
                if ode_type in ["General", "Separable", "Linear"]:
                    sol = sp.dsolve(eq, y)
                    st.subheader("Solution")
                    st.latex(sp.latex(sol))
                    if hasattr(sol, 'hint'):
                        st.write("Method Hint:", sol.hint)
                elif ode_type == "Bernoulli":
                    n_val = sp.sympify(n_input)
                    st.info("For Bernoulli equations, ensure the ODE is of the form: y'(t) = p(t)*y + q(t)*y**n")
                    st.write("Performing substitution: Let z = y^(1-n), then the ODE transforms to a linear equation in z.")
                    sol = sp.dsolve(eq, y)
                    st.subheader("General Solution (after substitution)")
                    st.latex(sp.latex(sol))
                elif ode_type == "Riccati (with known particular solution)":
                    if particular.strip() == "":
                        st.error("Please provide a known particular solution for the Riccati equation.")
                    else:
                        x1 = sp.sympify(particular, locals={'t': t})
                        st.info("Using the known particular solution, one can reduce the Riccati equation to a linear ODE.")
                        sol = sp.dsolve(eq, y)
                        st.subheader("General Solution (via Riccati transformation)")
                        st.latex(sp.latex(sol))
            except Exception as e:
                st.error("Error parsing or solving the differential equation: " + str(e))
    
    ##########################################
    # Integral Solver
    ##########################################
    elif prob_type == "Integral":
         st.markdown("### Integral Solver")
         st.markdown(
            """
Enter an integral expression to solve.
For example:  
`sin(t)**2`
            """
         )
         expr_input = st.text_input("Integral Expression", value="sin(t)**2", key="expr_input")
         var_input = st.text_input("Integration Variable", value="t", key="var_input")
        
         if st.button("Solve Integral", key="solve_integral"):
             try:
                 var = sp.symbols(var_input)
                 expr = sp.sympify(expr_input, locals={var_input: var})
                
                 # Compute the integral result
                 integral_result = sp.integrate(expr, var)
                 st.subheader("Integral Result")
                 st.latex(sp.latex(integral_result))
                
                 # Show manual integration steps if available
                 try:
                     steps = integral_steps(expr, var)
                     st.subheader("Integration Steps")
                     st.latex(steps.to_latex())
                 except Exception as step_e:
                     st.info("Step-by-step integration not available for this expression.")
             except Exception as e:
                 st.error("Error parsing or solving the integral: " + str(e))
    
    ##########################################
    # Equilibrium Analysis (with Stability)
    ##########################################
    elif prob_type == "Equilibrium Analysis":
         st.markdown("### Equilibrium Analysis for Autonomous ODE $x'(t) = v(x)$")
         st.markdown(
            r"""
Enter the function $v(x)$ and an initial condition $x_0$.  
Equilibrium points are found by solving $v(x)=0$.  
The time to reach an equilibrium from $x_0$ is estimated by evaluating  
$$
T = \int_{x_0}^{x_{eq}} \frac{dx}{v(x)}.
$$
Additionally, the stability of an equilibrium is determined by evaluating $v'(x_{eq})$:
- If $v'(x_{eq}) < 0$, the equilibrium is **stable**.
- If $v'(x_{eq}) > 0$, it is **unstable**.
            """
         )
         v_input = st.text_input("Enter $v(x)$", value="x*(1-x)", key="v_input")
         x0_input = st.text_input("Enter initial condition $x_0$", value="0.5", key="x0_input")
        
         st.subheader("Select Interval for Lipschitz Analysis")
         lip_choice = st.radio("Choose interval:",
                              ["Auto (based on $x_0$ and equilibrium points)", "Custom Interval", "Entire Real Line"],
                              key="lip_choice")
         if lip_choice == "Custom Interval":
             custom_lower = st.text_input("Enter lower bound", value="-10", key="lip_lower")
             custom_upper = st.text_input("Enter upper bound", value="10", key="lip_upper")
        
         if st.button("Analyze Equilibria", key="solve_equilibrium"):
             try:
                 x = sp.symbols('x')
                 v_expr = sp.sympify(v_input, locals={'x': x})
                 x0_val = sp.sympify(x0_input, locals={'x': x})
                
                 # Find equilibrium points
                 eq_points = sp.solve(v_expr, x)
                 st.subheader("Equilibrium Points")
                 if eq_points:
                     st.write(eq_points)
                     v_prime_expr = sp.diff(v_expr, x)
                     for eq_pt in eq_points:
                         derivative_val = v_prime_expr.subs(x, eq_pt)
                         try:
                             derivative_val_numeric = float(derivative_val.evalf())
                             stability = "stable" if derivative_val_numeric < 0 else "unstable" if derivative_val_numeric > 0 else "inconclusive"
                         except Exception:
                             stability = "inconclusive"
                         st.write(f"At equilibrium {eq_pt}: v'(x) = {derivative_val} -> {stability}")
                 else:
                     st.write("No equilibrium points found.")
                
                 # Compute time to reach equilibrium
                 st.subheader("Time to Reach Equilibrium (from $x_0$)")
                 for eq_pt in eq_points:
                     if sp.simplify(eq_pt - x0_val) == 0:
                         st.write(f"At $x_0 = {x0_val}$: already at equilibrium.")
                         continue
                     try:
                         T_expr = sp.integrate(1/v_expr, (x, x0_val, eq_pt))
                         if T_expr.is_finite:
                             st.write(f"Time to reach equilibrium at {eq_pt}:")
                             st.latex(sp.latex(T_expr))
                         else:
                             st.write(f"The time to reach equilibrium at {eq_pt} is infinite (divergent integral).")
                     except Exception as int_e:
                         st.write(f"Could not compute time to reach equilibrium at {eq_pt}: {int_e}")
                
                 # Plot v(x) and mark equilibria
                 numeric_eq = []
                 for eq_pt in eq_points:
                     try:
                         numeric_eq.append(float(eq_pt.evalf()))
                     except Exception:
                         pass
                 if numeric_eq:
                     x_min_auto = min(numeric_eq + [float(x0_val)]) - 1
                     x_max_auto = max(numeric_eq + [float(x0_val)]) + 1
                 else:
                     x_min_auto = float(x0_val) - 10
                     x_max_auto = float(x0_val) + 10
                
                 st.subheader("Graph of $v(x)$ with Equilibrium Points")
                 x_vals = np.linspace(x_min_auto, x_max_auto, 400)
                 v_func = sp.lambdify(x, v_expr, modules=['numpy'])
                 y_vals = v_func(x_vals)
                 fig, ax = plt.subplots()
                 ax.plot(x_vals, y_vals, label=r"$v(x)$")
                 ax.axhline(0, color='black', lw=0.5)
                 for eq_pt in eq_points:
                     try:
                         eq_numeric = float(eq_pt.evalf())
                         ax.plot(eq_numeric, 0, 'ro', label=f"Equilibrium: {eq_pt}")
                     except Exception:
                         pass
                 ax.set_xlabel("x")
                 ax.set_ylabel("v(x)")
                 ax.legend()
                 st.pyplot(fig)
                
                 # Lipschitz continuity analysis
                 st.subheader("Lipschitz Continuity Analysis")
                 v_prime_expr = sp.diff(v_expr, x)
                 st.write("The derivative $v'(x)$ is:")
                 st.latex(sp.latex(v_prime_expr))
                 if lip_choice == "Custom Interval":
                     try:
                         x_min_lip = float(sp.N(custom_lower))
                         x_max_lip = float(sp.N(custom_upper))
                     except Exception:
                         st.error("Invalid custom interval bounds. Reverting to auto interval.")
                         x_min_lip = x_min_auto
                         x_max_lip = x_max_auto
                 elif lip_choice == "Entire Real Line":
                     st.info("Analyzing over ℝ (approximation).")
                     x_min_lip = -1000
                     x_max_lip = 1000
                 else:
                     x_min_lip = x_min_auto
                     x_max_lip = x_max_auto
                
                 v_prime_func = sp.lambdify(x, v_prime_expr, modules=['numpy'])
                 x_vals_lip = np.linspace(x_min_lip, x_max_lip, 400)
                 v_prime_vals = np.abs(v_prime_func(x_vals_lip))
                 Lipschitz_const = np.max(v_prime_vals)
                 st.write("Estimated Lipschitz constant on [{}, {}]:".format(round(x_min_lip,2), round(x_max_lip,2)))
                 st.latex(sp.latex(Lipschitz_const))
                 st.write("Thus, $v(x)$ is Lipschitz continuous on this interval with Lipschitz constant L.")
                
                 st.subheader("Graph of $v'(x)$")
                 fig2, ax2 = plt.subplots()
                 ax2.plot(x_vals_lip, v_prime_func(x_vals_lip), color='green', label=r"$v'(x)$")
                 ax2.axhline(0, color='black', lw=0.5)
                 ax2.set_xlabel("x")
                 ax2.set_ylabel(r"$v'(x)$")
                 ax2.legend()
                 st.pyplot(fig2)
             except Exception as e:
                 st.error("Error in equilibrium analysis: " + str(e))
    
    ##########################################
    # Slope Field Visualization
    ##########################################
    elif prob_type == "Slope Field":
         st.markdown("### Slope Field Visualization for ODE $y'(t) = f(t, y)$")
         st.markdown(
            r"""
Enter the function $f(t, y)$ to plot its slope field.
For example:  
`sin(t) - y`
            """
         )
         f_input = st.text_input("Enter $f(t, y)$", value="sin(t) - y", key="f_input")
         t_min = st.number_input("Enter t minimum", value=0.0, key="t_min")
         t_max = st.number_input("Enter t maximum", value=10.0, key="t_max")
         y_min = st.number_input("Enter y minimum", value=-5.0, key="y_min")
         y_max = st.number_input("Enter y maximum", value=5.0, key="y_max")
         density = st.slider("Arrow density", 10, 50, 20, key="density")
         if st.button("Plot Slope Field", key="plot_slope_field"):
             try:
                 t_sym, y_sym = sp.symbols('t y')
                 f_expr = sp.sympify(f_input, locals={'t': t_sym, 'y': y_sym})
                 t_vals = np.linspace(t_min, t_max, density)
                 y_vals = np.linspace(y_min, y_max, density)
                 T, Y = np.meshgrid(t_vals, y_vals)
                 f_func = sp.lambdify((t_sym, y_sym), f_expr, modules=['numpy'])
                 U = np.ones_like(T)
                 V = f_func(T, Y)
                 N = np.sqrt(U**2 + V**2)
                 U_norm = U/N
                 V_norm = V/N
                 fig, ax = plt.subplots()
                 ax.quiver(T, Y, U_norm, V_norm, angles='xy')
                 ax.set_xlabel('t')
                 ax.set_ylabel('y')
                 ax.set_title("Slope Field for y' = " + f_input)
                 st.pyplot(fig)
             except Exception as e:
                 st.error("Error plotting slope field: " + str(e))

###############################################
# TAB 3: Notes on Differential Equations and Related Topics
###############################################
with tab3:
    st.title("Notes on Differential Equations and Dynamical Systems")
    notes_md = r"""
**First Order Linear Differential Equations**

For a simple equation 
$$
y'(x)=f(x),
$$
the solution can be computed by directly taking the integral
$$
y(x)=y(x_0)+\int_{x_0}^{x} f(z)\,dz.
$$

A separable equation is of the form 
$$
f(y)y' = g(x),
$$
so that if $F$ and $G$ are antiderivatives of $f$ and $g$ respectively, then
$$
\frac{d}{dx} F(y(x)) = g(x)
$$
and hence
$$
F(y(x)) = G(x) + C.
$$

**Autonomous and Non-Autonomous Systems**

Given a continuous function $v: U\subset \mathbb{R}^n \to \mathbb{R}^n$, a general first order autonomous system is
$$
x'(t)=v(x(t)),
$$
while a first order non-autonomous system is
$$
x'(t)=v(t,x(t)).
$$
Solutions $x(t)$ are called the *integral curves* of the vector field $v(x)$.

**First Order Linear Equations**

A general first order linear equation is of the form
$$
x'(t)=p(t)x(t)+q(t).
$$
After finding an antiderivative $P(t)$ with $P'(t)=p(t)$, one obtains
$$
\frac{d}{dt}\left(e^{-P(t)}x(t)\right)=q(t)e^{-P(t)}.
$$
Thus,
$$
x(t)e^{-P(t)}-x(t_0)e^{-P(t_0)}=\int_{t_0}^{t} e^{-P(s)}q(s)\,ds,
$$
or equivalently,
$$
x(t)=e^{P(t)-P(t_0)}x(t_0)+\int_{t_0}^{t} e^{P(t)-P(s)}q(s)\,ds,
$$
which can also be written as
$$
x(t)=e^{\int_{t_0}^{t} p(s)\,ds}x(t_0)+\int_{t_0}^{t} e^{\int_{s}^{t} p(r)\,dr}q(s)\,ds.
$$
If $p$ and $q$ are continuous on $(a,b)$, the solution is unique for a given $x(t_0)$.

**Bernoulli Equations**

A Bernoulli equation has the form
$$
x'(t)=p(t)x(t)+q(t)x^n(t).
$$
By making the substitution
$$
y(t)=x^{\,1-n}(t),
$$
the equation can be transformed into a linear equation for $y(t)$.

**Logistic Equation**

To model population growth, the logistic equation is often used:
$$
x'(t)=r(x)x,
$$
where $r(x)$ is a population-dependent proportionality factor.

**Ricatti Equations**

A Ricatti equation is of the form
$$
x'(t)=p(t)+q(t)x(t)+r(t)x^2(t).
$$
Although this equation cannot be solved explicitly in general, if one particular solution $x_1(t)$ is known, the general solution can be expressed as
$$
x(t)=x_1(t)+u(t),
$$
where $u(t)$ satisfies an equation that (after rearrangement) takes on the form of a Bernoulli equation:
$$
u'(t)=\big(q(t)+2r(t)x_1(t)\big)u(t)+r(t)u^2(t).
$$
A useful guess for $x_1(t)$ is often of the form $ct^\alpha$.

**Reduction of Order**

For a second order differential equation that depends only on $t$, $x'$, and $x''$, one may use the substitution $y=x'$ to reduce the order. In cases where the equation depends on $x$ rather than $t$, define 
$$
x(t)=(x(t), y(t))
$$
and
$$
v(x,y)=(y, f(x,y)).
$$
Then the solution satisfies
$$
x'(t)=v(x(t)).
$$
Assuming $y=x'$ is a function of $x$, by the chain rule
$$
x''=\frac{dy}{dt}=\frac{dy}{dx}\frac{dx}{dt}=y\frac{dy}{dx}=f(x,y).
$$

**Autonomous Systems and Flow Transformations**

For an autonomous ODE
$$
x'(t)=v(x(t)),
$$
Barrow’s formula gives
$$
t(x)=t_0+\int_{x_0}^{x} \frac{dz}{v(z)}.
$$
If $v$ is Lipschitz continuous, solutions are unique on the maximal interval.  
For the flow transformation associated with the IVP $x' = p(t)x + q(t)$, if $x(t_0)=x_0$ then
$$
\Phi_{t_1,t_0}(x_0)=x(t_1),
$$
with the composition property
$$
\Phi_{t_2,t_1}\big(\Phi_{t_1,t_0}(x)\big)=\Phi_{t_2,t_0}(x).
$$
The sensitivity function is given by
$$
\frac{d}{dx}\Phi_{t_1,t_0}(x)=e^{\int_{t_0}^{t_1} p(s)\,ds}.
$$

**Linear Systems**

For a linear vector field $v(x)=Ax$, a general solution is represented as a superposition of linearly independent solutions. If $\{v_1,\dots,v_n\}$ are eigenvectors of $A$ with eigenvalues $\mu_1,\dots,\mu_n$, then one can write
$$
M(t)=\Big[e^{t\mu_1}v_1,\dots,e^{t\mu_n}v_n\Big],
$$
and a solution with $x(t_0)=x_0$ is given by
$$
x(t)=M(t)M(0)^{-1}x_0.
$$
A flow on the solution is defined as
$$
\Psi_{t,s}(x)=M(t)M(s)^{-1}x.
$$
If all eigenvalues have strictly negative real parts, the solution converges to 0.

**Duhamel’s Formula**

For the non-homogeneous equation
$$
x'(t)=Ax(t)+g(t),
$$
the solution is given by
$$x
x(t)=e^{(t-t_0)A}x_0+\int_{t_0}^{t}e^{(t-s)A}g(s)\,ds.
$$

**Stability**

An equilibrium point $x^*$ of a vector field $v$ is:
- *Lyapunov stable* if for every $\varepsilon>0$ there exists $\delta>0$ such that any solution starting within $\delta$ of $x^*$ remains within $\varepsilon$ for all future time.
- *Asymptotically stable* if, in addition, solutions starting within $\delta$ converge to $x^*$ as $t\to\infty$.

When all eigenvalues of the linearization have strictly negative real parts, the equilibrium is asymptotically stable.

---

These notes summarize key methods and concepts in solving differential equations and analyzing dynamical systems.
    """
    st.markdown(notes_md, unsafe_allow_html=True)
