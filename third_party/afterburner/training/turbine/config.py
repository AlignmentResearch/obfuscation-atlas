import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.config import JobConfig


def model_args_and_train_spec_from_job_config(job_config: JobConfig):
    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)
    return model_args, train_spec
