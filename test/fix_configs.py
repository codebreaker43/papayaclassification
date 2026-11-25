import json
import os

def fix_file(path, is_json=True):
    print(f"Fixing {path}...")
    try:
        # Try reading as UTF-16 (common PowerShell output encoding)
        with open(path, 'r', encoding='utf-16') as f:
            content = f.read()
    except:
        try:
            # Try UTF-8 if UTF-16 fails
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            return

    # Find the start of the JSON or XML
    if is_json:
        start = content.find('{')
        end = content.rfind('}') + 1
    else: # XML/Plist
        start = content.find('<?xml')
        end = content.rfind('</plist>') + 8

    if start != -1 and end != -1:
        clean_content = content[start:end]
        
        # Verify it's valid JSON if supposed to be
        if is_json:
            try:
                json.loads(clean_content)
                print("JSON is valid.")
            except json.JSONDecodeError as e:
                print(f"JSON validation failed: {e}")
                # Try simple cleanup of common garbage if validation fails
                clean_content = clean_content.strip()

        # Write back as clean UTF-8
        with open(path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        print(f"Saved cleaned file to {path}")
    else:
        print(f"Could not find valid start/end markers in {path}")

# Fix Android
fix_file('android/app/google-services.json', is_json=True)

# Fix iOS
fix_file('ios/Runner/GoogleService-Info.plist', is_json=False)
