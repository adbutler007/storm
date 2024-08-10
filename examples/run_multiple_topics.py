import json
import subprocess
import os

# Read the JSON file
with open('examples/topics.json', 'r') as f:
    data = json.load(f)

topics = data['topics']

# Ensure the output directory exists
base_output_dir = './results/claude_multi_topic'
os.makedirs(base_output_dir, exist_ok=True)

# Run STORM for each topic
for topic in topics:
    print(f"Processing topic: {topic}")
    
    # Create a safe filename from the topic
    safe_topic = ''.join(c if c.isalnum() else '_' for c in topic)[:50]
    
    # Create a topic-specific output directory
    topic_output_dir = os.path.join(base_output_dir, safe_topic)
    
    # Run the STORM pipeline
    command = [
        'python', 'examples/run_storm_wiki_gpt_with_VectorRM.py'
    ]
    
    # Use subprocess to run the command and capture output
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Provide the topic as input
    stdout, stderr = process.communicate(input=f"{topic}\n")
    
    # Print the output and errors (if any)
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)
    
    print(f"Finished processing topic: {topic}\n")

print("All topics processed.")