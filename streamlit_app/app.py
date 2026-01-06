import streamlit as st

def app():
    """"My First ST app"""
    
    st.title("RAG from Scratch")
    st.header("Streamlit Application")
    st.subheader("Welcome!")
    st.write("Welcome to the RAG from Scratch Streamlit App!")
    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
    )
    left_column, a, b, c, right_column = st.columns(5)
    # You can use a column just like st.sidebar:
    left_column.button('Press me!')
    a.write('hi')
    with right_column:
        chosen = st.radio(
            'Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
        st.write(f"You are in {chosen} house!")
if __name__ == "__main__":
    app()