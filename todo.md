
[x] integrate preference prior + gaze-conditioned diffusion's relevant code
  [x] update training +
  [x] adapter architecture
[x] append adapter-processed semantic embeds to set pre-cached prompts (better initialization)

##### Experiments:
[] lora, lr, etc.
[] ablating a user ID conditioning with preferrence history conditioning
  - if performance is comparable, we've "solved" the cold start problem

##### Features:
[] re-enable demographics conditioning
[] integrate Gabriel's rebeca quant eval

[] condition on 
  [x] scores + image embeds
    - can likely improve design here
  [] text prompt
    - ensure text embeds come last so 0th first embed is always providing target score
  [] demographics

[] val should show inputted images + their scores

[] qual eval
  [] pad & drop so that we could give just e.g. 2 images instead of four

##### Codebase improvements
[] stash current .py files into logs each training run
[] avoid storing e.g. both model.config.seed and model.seed

