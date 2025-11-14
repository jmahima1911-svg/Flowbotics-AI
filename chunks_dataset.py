"""
Document Chunking Script for Gen AI Chatbot
Uses LangChain (free, open-source) for intelligent text splitting
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_markdown_files(directory_path, chunk_size=1000, chunk_overlap=200):
    """
    Chunk all markdown files in the dataset directory
    
    Args:
        directory_path: Path to your chatbot_dataset folder
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between chunks to maintain context
    
    Returns:
        List of document chunks with metadata
    """
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split by paragraphs, then sentences
    )
    
    all_chunks = []
    
    # List of your markdown files
    files = [
        "customer_success_stories.md",
        "flowbotics_services.md",
        "frequently_asked_questions.md",
        "integration_guides.md",
        "pricing_packages.md",
        "technical_specifications.md"
    ]
    
    # Process each file
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into chunks
            chunks = text_splitter.split_text(content)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'content': chunk,
                    'source': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
            
            print(f"✓ {filename}: {len(chunks)} chunks created")
            
        except FileNotFoundError:
            print(f"✗ {filename}: File not found")
        except Exception as e:
            print(f"✗ {filename}: Error - {str(e)}")
    
    return all_chunks


def save_chunks(chunks, output_file="chunked_data.txt"):
    """Save chunks to a text file for inspection"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n{'='*80}\n")
            # Handle both dict and string chunks
            if isinstance(chunk, dict):
                source = chunk.get('source', 'unknown')
                chunk_id = chunk.get('chunk_id', i)
                total = chunk.get('total_chunks', len(chunks))
                content = chunk.get('content', str(chunk))
                f.write(f"Source: {source} | Chunk: {chunk_id+1}/{total}\n")
            else:
                content = str(chunk)
                f.write(f"Chunk: {i+1}/{len(chunks)}\n")
            f.write(f"{'='*80}\n")
            f.write(content)
            f.write(f"\n")
    print(f"\n✓ Chunks saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Set your dataset directory path
    dataset_path = "./chatbot_dataset"
    
    # Chunk the documents
    chunks = chunk_markdown_files(
        directory_path=dataset_path,
        chunk_size=1000,      # Adjust based on your needs
        chunk_overlap=200     # Maintains context between chunks
    )
    
    print(f"\n📊 Total chunks created: {len(chunks)}")
    
    # Save chunks for inspection
    save_chunks(chunks)
    
    # Display sample chunk
    if chunks:
        print("\n" + "="*80)
        print("SAMPLE CHUNK:")
        print("="*80)
        print(f"Source: {chunks[0]['source']}")
        print(f"Chunk: {chunks[0]['chunk_id']+1}/{chunks[0]['total_chunks']}")
        print("-"*80)
        print(chunks[0]['content'][:500] + "...")