import os
from googleapiclient.discovery import build

def shutdown_vm(request):
    project = os.getenv("PROJECT_ID")
    zone = os.getenv("ZONE")
    instance = os.getenv("VM_NAME")

    compute = build('compute', 'v1')

    try:
        request = compute.instances().stop(project=project, zone=zone, instance=instance)
        response = request.execute()
        return f"VM {instance} in zone {zone} has been stopped successfully."
    except Exception as e:
        return f"Failed to stop VM {instance}: {str(e)}"
