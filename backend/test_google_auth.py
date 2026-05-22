"""Quick test script to verify Google Cloud authentication and Vertex AI access."""

import json
import os
import sys

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "google-credentials.json")


def main():
    # 1. Load credentials file
    print("Step 1: Loading google-credentials.json ...")
    try:
        with open(CREDENTIALS_PATH) as f:
            creds_data = json.load(f)
        project_id = creds_data.get("project_id", "unknown")
        client_email = creds_data.get("client_email", "unknown")
        print(f"  ✓ Loaded credentials for: {client_email}")
        print(f"  ✓ Project ID: {project_id}")
    except FileNotFoundError:
        print(f"  ERROR: File not found at {CREDENTIALS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON — {e}")
        sys.exit(1)

    # 2. Authenticate with Google Cloud
    print("\nStep 2: Authenticating with Google Cloud ...")
    try:
        import google.auth
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        print(f"  ✓ Authenticated as: {credentials.service_account_email}")
    except Exception as e:
        print(f"  ERROR: Authentication failed — {e}")
        sys.exit(1)

    # 3. List available Vertex AI models
    print("\nStep 3: Listing Vertex AI models ...")
    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project_id, credentials=credentials, location="us-central1")

        models = aiplatform.Model.list()
        if models:
            print(f"  ✓ Found {len(models)} model(s):")
            for m in models[:5]:  # show first 5
                print(f"    - {m.display_name}")
        else:
            print("  ✓ No custom models found (this is normal for a fresh project)")

    except Exception as e:
        print(f"  ERROR: Vertex AI listing failed — {e}")
        sys.exit(1)

    print("\n✓ SUCCESS — Google Cloud authentication and Vertex AI access confirmed.")


if __name__ == "__main__":
    main()
