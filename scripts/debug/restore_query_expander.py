import subprocess

try:
    # Run git show to get the committed version of utils/query_expander.py
    print("Reading HEAD:utils/query_expander.py...")
    content = subprocess.check_output(
        ["git", "show", "HEAD:utils/query_expander.py"],
        stderr=subprocess.STDOUT
    ).decode("utf-8")
    
    # Write it back to utils/query_expander.py
    with open("utils/query_expander.py", "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Successfully restored utils/query_expander.py content from HEAD!")
except subprocess.CalledProcessError as e:
    print(f"Error executing git show: {e.output.decode('utf-8')}")
except Exception as e:
    print(f"General error: {e}")
