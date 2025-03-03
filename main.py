import streamlit as st
import sympy as sp
from sympy.integrals.manualintegrate import integral_steps
import numpy as np
import matplotlib.pyplot as plt

# Create two tabs: one for matrix calculator, one for DEs & integrals.
tab1, tab2 = st.tabs(["Matrix Calculator", "Differential Equations & Integrals"])

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
            # Split by comma; if only one element is found, try splitting by whitespace
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
        
        # Calculate eigenvalues, multiplicities and eigenvectors (supports complex values)
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
        
        # Define the symbol t and compute the fundamental matrix M(t)=exp(At)
        t = sp.symbols('t', real=True)
        M_t = (A * t).exp()
        st.subheader("Fundamental Matrix \(M(t)=e^{At}\)")
        st.latex(sp.latex(M_t))
        
        # Compute M(0) and its inverse
        M0 = M_t.subs(t, 0)
        st.subheader("M(0) and its Inverse \(M(0)^{-1}\)")
        st.write("M(0):")
        st.latex(sp.latex(M0))
        try:
            M0_inv = M0.inv()
            st.write("M(0)\(^{-1}\):")
            st.latex(sp.latex(M0_inv))
        except Exception as e:
            st.write("M(0) is not invertible.")
        
        # Compute and display powers of the matrix: A^2 and A^3
        A2 = sp.simplify(A**2)
        A3 = sp.simplify(A**3)
        st.subheader("Matrix Powers")
        st.write("A\(^2\):")
        st.latex(sp.latex(A2))
        st.write("A\(^3\):")
        st.latex(sp.latex(A3))
        
        # Compute and display the inverse of A (if it exists)
        st.subheader("Inverse of Matrix A")
        try:
            A_inv = A.inv()
            st.latex(sp.latex(A_inv))
        except Exception as e:
            st.write("Matrix A is not invertible.")
        
        # Compute the Jordan form to extract the generalized eigenvectors.
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
- The **first column** in that block is a proper eigenvector.
- Any **subsequent columns** in the same block are generalized eigenvectors.
                """
            )
            
            # Also, show how the fundamental matrix can be constructed from the Jordan form:
            exp_Jt = sp.exp(J * t)
            M_t_jordan = sp.simplify(P * exp_Jt * P.inv())
            st.write("Fundamental Matrix computed via the Jordan decomposition, i.e.,")
            st.latex(sp.latex(sp.Eq(sp.symbols('M(t)'), P * sp.exp(J * t) * P.inv())))
            st.latex(sp.latex(M_t_jordan))
        except Exception as e:
            st.write("Could not compute the Jordan form and generalized eigenvectors. This might occur if the matrix is already diagonalizable or if an error occurred.")

#########################
# TAB 2: Differential Equations, Integrals & Equilibrium Analysis
#########################
with tab2:
    st.title("Differential Equations, Integrals & Equilibrium Analysis")
    
    # Create radio options for three functionalities
    prob_type = st.radio("Select Problem Type:", 
                         ["Differential Equation", "Integral", "Equilibrium Analysis"],
                         key="prob_type")
    
    if prob_type == "Differential Equation":
        st.markdown(
            """
**Enter a differential equation in terms of** `y(t)`  
For example:  
`y'(t) - y(t) = 0`
            """
        )
        de_input = st.text_input("Differential Equation", value="y'(t) - y(t) = 0", key="de_input")
        
        if st.button("Solve Differential Equation", key="solve_de"):
            try:
                # Define the variable and function
                t = sp.symbols('t')
                y = sp.Function('y')(t)
                
                # Parse the input; split at '=' if present
                parts = de_input.split('=')
                if len(parts) == 2:
                    lhs = sp.sympify(parts[0], locals={'y': y, 't': t})
                    rhs = sp.sympify(parts[1], locals={'y': y, 't': t})
                    eq = sp.Eq(lhs, rhs)
                else:
                    eq = sp.sympify(de_input, locals={'y': y, 't': t})
                
                # Solve the ODE
                sol = sp.dsolve(eq, y)
                st.subheader("Solution")
                st.latex(sp.latex(sol))
                # Display the method/hint if available
                if hasattr(sol, 'hint'):
                    st.write("Method Hint:", sol.hint)
            except Exception as e:
                st.error("Error parsing or solving the differential equation: " + str(e))
    
    elif prob_type == "Integral":
        st.markdown(
            """
**Enter an integral expression to solve**  
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
                if integral_steps:
                    try:
                        steps = integral_steps(expr, var)
                        st.subheader("Integration Steps")
                        steps_latex = steps.to_latex()
                        st.latex(steps_latex)
                    except Exception as step_e:
                        st.info("Step-by-step integration not available for this expression.")
                else:
                    st.info("Step-by-step integration functionality is not available in your version of Sympy.")
            except Exception as e:
                st.error("Error parsing or solving the integral: " + str(e))
    
    elif prob_type == "Equilibrium Analysis":
        st.markdown(
            r"""
**Equilibrium Analysis for Autonomous ODE \(x'(t) = v(x)\)**

Enter the function \(v(x)\) and an initial condition \(x_0\).  
Equilibrium points are found by solving \(v(x)=0\).  
The time to reach an equilibrium from \(x_0\) is estimated by evaluating  
\[
T = \int_{x_0}^{x_{eq}} \frac{dx}{v(x)}.
\]
If \(T\) is finite, the equilibrium is reached in finite time.
            """
        )
        v_input = st.text_input("Enter \(v(x)\)", value="x*(1-x)", key="v_input")
        x0_input = st.text_input("Enter initial condition \(x_0\)", value="0.5", key="x0_input")
        
        st.subheader("Select Interval for Lipschitz Analysis")
        lip_choice = st.radio("Choose interval:",
                              ["Auto (based on \(x_0\) and equilibrium points)", "Custom Interval", "Entire Real Line"],
                              key="lip_choice")
        if lip_choice == "Custom Interval":
            custom_lower = st.text_input("Enter lower bound", value="-10", key="lip_lower")
            custom_upper = st.text_input("Enter upper bound", value="10", key="lip_upper")
        
        if st.button("Analyze Equilibria", key="solve_equilibrium"):
            try:
                x = sp.symbols('x')
                v_expr = sp.sympify(v_input, locals={'x': x})
                x0_val = sp.sympify(x0_input, locals={'x': x})
                
                # Find equilibrium points by solving v(x)=0
                eq_points = sp.solve(v_expr, x)
                st.subheader("Equilibrium Points")
                if eq_points:
                    st.write(eq_points)
                else:
                    st.write("No equilibrium points found.")
                
                # For each equilibrium, compute the time T = ∫(dx/v(x)) from x0 to equilibrium
                st.subheader("Time to Reach Equilibrium (from \(x_0\))")
                for eq_pt in eq_points:
                    if sp.simplify(eq_pt - x0_val) == 0:
                        st.write(f"At \(x_0 = {x0_val}\): already at equilibrium.")
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
                
                # Determine a default (auto) plotting range: center around x₀ and include equilibria if numeric
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
                
                # Plot v(x) and mark the equilibrium points
                st.subheader("Graph of \(v(x)\) with Equilibrium Points")
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
                
                # --------------------------
                # Lipschitz Continuity Analysis
                # --------------------------
                st.subheader("Lipschitz Continuity Analysis")
                v_prime_expr = sp.diff(v_expr, x)
                st.write("The derivative \(v'(x)\) is:")
                st.latex(sp.latex(v_prime_expr))
                
                # Choose the interval for Lipschitz analysis based on the radio selection
                if lip_choice == "Custom Interval":
                    try:
                        x_min_lip = float(sp.N(custom_lower))
                        x_max_lip = float(sp.N(custom_upper))
                    except Exception:
                        st.error("Invalid custom interval bounds. Reverting to auto interval.")
                        x_min_lip = x_min_auto
                        x_max_lip = x_max_auto
                elif lip_choice == "Entire Real Line":
                    st.info("Analyzing over ℝ using numerical sampling over a large interval (approximation).")
                    x_min_lip = -1000
                    x_max_lip = 1000
                else:  # Auto
                    x_min_lip = x_min_auto
                    x_max_lip = x_max_auto
                
                v_prime_func = sp.lambdify(x, v_prime_expr, modules=['numpy'])
                x_vals_lip = np.linspace(x_min_lip, x_max_lip, 400)
                v_prime_vals = np.abs(v_prime_func(x_vals_lip))
                Lipschitz_const = np.max(v_prime_vals)
                st.write("The estimated Lipschitz constant on the interval [{}, {}] is:".format(round(x_min_lip,2), round(x_max_lip,2)))
                st.latex(sp.latex(Lipschitz_const))
                st.write("Thus, \(v(x)\) is Lipschitz continuous on this interval with Lipschitz constant L.")
                
                # --------------------------
                # Graph of v'(x)
                # --------------------------
                st.subheader("Graph of \(v'(x)\)")
                fig2, ax2 = plt.subplots()
                ax2.plot(x_vals_lip, v_prime_func(x_vals_lip), color='green', label=r"$v'(x)$")
                ax2.axhline(0, color='black', lw=0.5)
                ax2.set_xlabel("x")
                ax2.set_ylabel(r"$v'(x)$")
                ax2.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.error("Error in equilibrium analysis: " + str(e))
