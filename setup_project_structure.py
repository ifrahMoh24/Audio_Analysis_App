import os

def create_project_structure(base_path):
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "scripts",
        "notebooks",
        "static",
        "templates"
    ]

    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

    # Create initial files
    open(os.path.join(base_path, "app.py"), "w").close()
    open(os.path.join(base_path, "requirements.txt"), "w").close()
    open(os.path.join(base_path, "README.md"), "w").close()
    print("Initial files created: app.py, requirements.txt, README.md")

if __name__ == "__main__":
    base_path = "/Users/ifrahmohamed/Desktop/AudioAnalysisApp"  # Adjust the path as needed
    create_project_structure(base_path)
    print("Project structure setup completed!")
