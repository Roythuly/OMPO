<h1>OMPO</span></h1>

Official implementation of

`OMPO: A Unified Framework for RL under Policy and Dynamics Shifts` by

Yu Luo, Tianying Ji, Fuchun Sun, Jianwei Zhang, Huazhe Xu and Xianyuan Zhan

## Getting started

We provide examples on how to train and evaluate **OMPO** agent.

### Training

See below examples on how to train OBAC on a single task.

```python
python main_stationary.py --env_name YOUR_TASK
```

We recommend using default hyperparameters. See `utils/default_config.py` for a full list of arguments.

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@inproceedings{Luo2024ompo,
  title={OMPO: A Unified Framework for RL under Policy and Dynamics Shifts}, 
  author={Yu Luo and Tianjing Ji and Fuchun Sun and Jianwei Zhang and Huazhe Xu and Xianyuan Zhan},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

----

## Contributing

Please feel free to participate in our project by opening issues or sending pull requests for any enhancements or bug reports you might have. We’re striving to develop a codebase that’s easily expandable to different settings and tasks, and your feedback on how it’s working is greatly appreciated!

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
