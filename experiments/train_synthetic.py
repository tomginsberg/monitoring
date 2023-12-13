from training.train import Trainer
from data.synthetic import SyntheticData
import yaml

data = SyntheticData(n_samples=5000)
trainer = Trainer(
    dataset=data,
    training_criteria=('before_date', {'year': 2020, 'month': 1, 'day': 1}),
    input_features=['x1', 'x2'],
    label='y',
)
config = trainer.fit(output_path='models/synthetic.model')
del config['model']
print(
    yaml.dump(config)
)
