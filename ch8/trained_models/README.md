#### The models were saved using:

```python
torch.save({'actor_state_dict':self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict(), 'global_step_num': self.global_step_num, 'params':self.params, 'ep_reward': ep_reward}, "./trained_models/Pendulum-v0_best.ptm")
```
