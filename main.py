import streamlit as st
import sympy as sp

st.title("Matrix Calculator: Eigenvalues, Eigenvectors & Matrix Exponential")

st.markdown(
    """
Enter your matrix below. Each row should be on a new line with entries separated by commas or spaces.
For example:

-1, 0, 1

0, -2, 4

0,0,-2

You can enter fractions (e.g. `1/2`) if you check the "Use fractions" box.
    """
)

# Input area for the matrix
matrix_input = st.text_area("Matrix Input", value="-1, 0, 1\n0, -2, 4\n0,0,-2", height=150)
use_fractions = st.checkbox("Use fractions (symbolic)", value=True)

if st.button("Calculate"):
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
                    # Convert the input to a Rational number (fraction)
                    val = sp.Rational(item)
                except Exception:
                    # Fallback: simplify a float
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
    
    # Calculate eigenvalues and eigenvectors (supports complex values)
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
