import os
import sys
import subprocess

def main():
    # Check if this is first run
    if not os.path.exists('.venv'):
        print("First run detected. Installing dependencies...")
        install_script = os.path.join(os.path.dirname(__file__), 'install_mdm.sh')
        subprocess.run(['chmod', '+x', install_script])
        subprocess.run([install_script])

    # Now run the actual Flask app
    from backend.main import app
    app.run(port=3000)

if __name__ == '__main__':
    main() 