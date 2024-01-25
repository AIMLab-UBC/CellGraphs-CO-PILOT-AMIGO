import neptune.new as neptune
import wandb

import utils.distributed


class Monitoring:

    def __init__(self, config):
        self.wandb, self.neptune = None, None
        if not utils.distributed.is_main_process():
            return None

        name = config['job_name']
        if config['fold_number'] is not None:
            name += f"_{config['fold_number']}"
        name += f"_seed{config['seed']}"

        self.neptune = None
        if config['neptune']:
            self.neptune = neptune.init(name=name, project='GNN', source_files='**/*.py')

        self.wandb = None
        if config['wandb']:
            self.wandb = wandb
            wandb.init(project='GNN', config=config, group=config['job_name'], name=name)
            wandb.run.log_code('.', exclude_fn=lambda path: 'wandb' in path)

        self.tags(config["tags"])

    def get_run_name(self):
        if self.wandb:
            return self.wandb.run.name
        return None

    def save_file(self, file):
        if self.wandb:
            self.wandb.save(file)

    def save_df(self, df, key='predictions', step=None):
        if self.wandb:
            table = self.wandb.Table(dataframe=df)
            self.wandb.log({key: table}, step=step)

    def tags(self, tags):
        if not self.wandb:
            return
        if tags is None:
            return
        if not isinstance(tags, tuple):
            tags = tuple(tags)
        self.wandb.run.tags += tags

    def log(self, metrics, step=None):
        if not utils.distributed.is_main_process():
            return

        if self.wandb:
            self.wandb.log(metrics, step=step)

        for k, v in metrics.items():
            if self.neptune:
                self.neptune[k].log(v)
