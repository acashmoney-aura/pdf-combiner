import streamlit as st
import PyPDF2
import io
import base64
from PIL import Image
import tempfile
import os
import pdf2image
import uuid
import time
from functools import lru_cache

# Set page configuration
st.set_page_config(page_title="PDF Combiner & Editor", layout="wide")

# Initialize session state variables if they don't exist
if 'pages' not in st.session_state:
    st.session_state.pages = []  # Will store (page_content, original_pdf_name, page_number, rotation)
if 'page_order' not in st.session_state:
    st.session_state.page_order = []  # Will store indices to st.session_state.pages
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()  # Keep track of already processed files
if 'thumbnails' not in st.session_state:
    st.session_state.thumbnails = {}  # Cache for thumbnails

# Cache the thumbnail creation function
@st.cache_data
def cached_create_thumbnail(page_id, rotation=0):
    """Create a cached thumbnail image of a PDF page"""
    page_bytes, _, _, _ = st.session_state.pages[page_id]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(page_bytes.getvalue())
        tmp_path = tmp_file.name
    
    # Convert PDF page to image
    images = pdf2image.convert_from_path(tmp_path, size=(150, 200))
    img = images[0]
    
    # Apply rotation if needed
    if rotation != 0:
        img = img.rotate(rotation, expand=True)
    
    # Convert PIL image to bytes for display
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    # Clean up
    os.unlink(tmp_path)
    
    return buf

def extract_pages_from_pdf(uploaded_file):
    """Extract pages from an uploaded PDF file"""
    pages = []
    
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Open the PDF file
    with open(tmp_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        
        # Extract each page
        for i in range(len(pdf_reader.pages)):
            # Create a new PDF writer for this single page
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[i])
            
            # Save the single page to a byte stream
            page_bytes = io.BytesIO()
            pdf_writer.write(page_bytes)
            page_bytes.seek(0)
            
            # Save the page content and metadata
            pages.append((page_bytes, uploaded_file.name, i + 1, 0))  # 0 degrees rotation initially
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return pages

def combine_pdfs():
    """Combine all pages in the current order into a single PDF"""
    merger = PyPDF2.PdfMerger()
    
    for idx in st.session_state.page_order:
        page_bytes, _, _, rotation = st.session_state.pages[idx]
        
        # Create temporary file for the page
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(page_bytes.getvalue())
            tmp_path = tmp_file.name
        
        # Open the PDF file and apply rotation if needed
        with open(tmp_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            writer = PyPDF2.PdfWriter()
            page = reader.pages[0]
            
            if rotation == 90:
                page.rotate(90)
            elif rotation == 180:
                page.rotate(180)
            elif rotation == 270:
                page.rotate(270)
            
            writer.add_page(page)
            rotated_bytes = io.BytesIO()
            writer.write(rotated_bytes)
            rotated_bytes.seek(0)
            
            # Add to merger
            merger.append(fileobj=rotated_bytes)
        
        # Clean up
        os.unlink(tmp_path)
    
    # Save the combined PDF to a bytes buffer
    output = io.BytesIO()
    merger.write(output)
    merger.close()
    output.seek(0)
    
    return output

def get_download_link(pdf_bytes, filename="combined.pdf"):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download Combined PDF</a>'
    return href

def reset_app():
    """Reset the application to its initial state"""
    st.session_state.pages = []
    st.session_state.page_order = []
    st.session_state.processed_files = set()
    st.session_state.thumbnails = {}
    st.cache_data.clear()

def rotate_page(idx):
    """Rotate the specified page by 90 degrees clockwise"""
    page_bytes, pdf_name, page_num, current_rotation = st.session_state.pages[idx]
    # Update to the next rotation (0 -> 90 -> 180 -> 270 -> 0)
    new_rotation = (current_rotation + 90) % 360
    st.session_state.pages[idx] = (page_bytes, pdf_name, page_num, new_rotation)
    
    # Update the thumbnail in cache
    thumbnail_key = f"{idx}_{new_rotation}"
    if thumbnail_key not in st.session_state.thumbnails:
        st.session_state.thumbnails[thumbnail_key] = cached_create_thumbnail(idx, new_rotation)

def move_page_up(idx):
    """Move a page up in the order"""
    if idx > 0:
        st.session_state.page_order[idx], st.session_state.page_order[idx-1] = st.session_state.page_order[idx-1], st.session_state.page_order[idx]

def move_page_down(idx):
    """Move a page down in the order"""
    if idx < len(st.session_state.page_order) - 1:
        st.session_state.page_order[idx], st.session_state.page_order[idx+1] = st.session_state.page_order[idx+1], st.session_state.page_order[idx]

def remove_page(idx):
    """Remove a page from the order"""
    st.session_state.page_order.pop(idx)

# App title and description
st.title("PDF Combiner & Editor")
st.write("Upload PDFs, combine them, reorder pages, delete pages, and rotate as needed.")

# File uploader section
with st.expander("Upload PDF Files", expanded=True):
    uploaded_files = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if this file has already been processed by checking filename and size
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if file_id not in st.session_state.processed_files:
                # Show processing status
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Extract pages from the PDF
                    new_pages = extract_pages_from_pdf(uploaded_file)
                    
                    # Add new pages to the session state
                    start_idx = len(st.session_state.pages)
                    st.session_state.pages.extend(new_pages)
                    
                    # Add indices of new pages to the page order
                    st.session_state.page_order.extend(range(start_idx, start_idx + len(new_pages)))
                    
                    # Precalculate thumbnails for new pages
                    for i, page_idx in enumerate(range(start_idx, start_idx + len(new_pages))):
                        thumbnail_key = f"{page_idx}_0"  # Initial rotation is 0
                        st.session_state.thumbnails[thumbnail_key] = cached_create_thumbnail(page_idx)
                    
                    # Mark this file as processed
                    st.session_state.processed_files.add(file_id)
                
                st.success(f"Added {len(new_pages)} pages from {uploaded_file.name}")

# Display and interact with pages
if st.session_state.pages:
    st.write("---")
    
    # Add container for page management
    with st.container():
        st.subheader("Manage Pages")
        st.write("You can reorder, delete, or rotate pages. Click 'Rotate' to switch between portrait and landscape.")
        
        # Create columns for the page grid
        cols_per_row = 4
        num_pages = len(st.session_state.page_order)
        
        for i in range(0, num_pages, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                idx = i + j
                if idx < num_pages:
                    page_idx = st.session_state.page_order[idx]
                    page_bytes, pdf_name, page_num, rotation = st.session_state.pages[page_idx]
                    
                    with cols[j]:
                        # Get cached thumbnail
                        thumbnail_key = f"{page_idx}_{rotation}"
                        if thumbnail_key not in st.session_state.thumbnails:
                            st.session_state.thumbnails[thumbnail_key] = cached_create_thumbnail(page_idx, rotation)
                        
                        # Display page thumbnail
                        st.image(st.session_state.thumbnails[thumbnail_key], caption=f"{pdf_name} - Page {page_num}")
                        
                        # Add buttons for page operations with unique keys
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if st.button("↑", key=f"up_{idx}_{page_idx}", disabled=(idx == 0)):
                                move_page_up(idx)
                                st.rerun()
                        
                        with col2:
                            if st.button("↓", key=f"down_{idx}_{page_idx}", disabled=(idx == num_pages - 1)):
                                move_page_down(idx)
                                st.rerun()
                        
                        with col3:
                            if st.button("🔄", key=f"rotate_{idx}_{page_idx}"):
                                rotate_page(page_idx)
                                st.rerun()
                        
                        with col4:
                            if st.button("🗑️", key=f"delete_{idx}_{page_idx}"):
                                remove_page(idx)
                                st.rerun()
        
        st.write("---")
        
        # Action buttons in columns
        col1, col2 = st.columns(2)
        
        # Generate and display download button
        with col1:
            if st.button("Combine PDFs", key="combine_button", disabled=(len(st.session_state.page_order) == 0)):
                with st.spinner("Combining PDFs..."):
                    combined_pdf = combine_pdfs()
                st.success("PDFs combined successfully!")
                st.markdown(get_download_link(combined_pdf), unsafe_allow_html=True)
        
        # Reset app button
        with col2:
            if st.button("Clear All", key="clear_button"):
                reset_app()
                st.cache_data.clear()
                st.rerun()

else:
    st.info("Upload PDF files to get started.")

# Add requirements info
st.write("---")
st.caption("Note: This app requires the following Python packages: streamlit, PyPDF2, pdf2image, and Pillow.")
st.caption("For pdf2image to work, you need to install poppler. On Ubuntu/Debian: 'apt-get install poppler-utils', on macOS: 'brew install poppler', on Windows: download and install from the poppler website.")