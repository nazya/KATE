import os
from mlflow import MlflowClient
from mlflow.entities import ViewType


class MLFlowLogger():
    def __init__(self):
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.c = MlflowClient(tracking_uri=tracking_uri)
        self.e = self.c.get_experiment_by_name(experiment_name)
        self.e_id = self.c.create_experiment(experiment_name)\
            if self.e is None else self.e.experiment_id

        self.enabled = True

    def check_exist(self, config):
        check_exist = eval(os.environ['MLFLOW_CHECK_EXIST'])
        if not check_exist:
            return False

        if self.e is not None:
            filter_string = list()
            for key in config.keys():
                if isinstance(config[key], dict):
                    value = config[key]['name']
                else:
                    value = config[key]
                filter_string.append(f'params.{key}="{value}"')
            filter_string.append('attributes.status="FINISHED"')
            tags = eval(os.environ['MLFLOW_RUN_TAGS'])
            for key, value in tags.items():
                filter_string.append(f'tags."{key}" = "{value}"')
            # print(f"{filter_string=}")
            filter_string = ' and '.join(filter_string)
            runs = self.c.search_runs(experiment_ids=[self.e.experiment_id],
                                      filter_string=filter_string,
                                      run_view_type=ViewType.ACTIVE_ONLY)
            if len(runs):
                return True
        return False

    def init(self, config):
        if not self.enabled:
            return

        r = self.c.create_run(experiment_id=self.e_id,
                              run_name=os.environ['MLFLOW_RUN_NAME'],
                              tags=eval(os.environ['MLFLOW_RUN_TAGS']))
        self.r_id = r.info.run_id
        self.c.log_dict(self.r_id, config, 'config.json')
        for key, value in config.items():
            if isinstance(value, dict):
                value = value['name']
            self.c.log_param(self.r_id, key, value)

    def log_metrics(self, metrics, step):
        if not self.enabled:
            return

        for key in metrics.keys():
            self.c.log_metric(self.r_id, key, metrics[key], step=step)
        print('Step ' + '{}: '.format(step).rjust(6) +
              ' '.join("{}: {:.5f}".format(k, v) for k, v in metrics.items() if 'weights' not in k), end='\r', flush=True)

    def terminate(self):
        if not self.enabled:
            return

        self.c.set_terminated(self.r_id)
        print()