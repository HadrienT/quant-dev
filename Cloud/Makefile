# Define variables for Terraform
TF_PLAN=tf.plan
TF_VARS=quantdev.tfvars
SECRETS_VARS=secrets.tfvars

# Default target
.PHONY: plan
plan:
	terraform plan --out=$(TF_PLAN) --var-file=$(TF_VARS) --var-file=$(SECRETS_VARS)

# Optional: Add a target to apply the plan
.PHONY: apply
apply:
	terraform apply $(TF_PLAN) 

# Optional: Add a target to destroy resources
.PHONY: destroy
destroy:
	terraform destroy --var-file=$(TF_VARS) --var-file=$(SECRETS_VARS)

# Optional: Add a target to initialize Terraform
.PHONY: init
init:
	terraform init
