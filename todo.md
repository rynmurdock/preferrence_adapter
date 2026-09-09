
[x] integrate preference prior + gaze-conditioned diffusion's relevant code
  [x] update training +
  [x] adapter architecture
[] re-enable demographics conditioning

[] integrate Gabriel's rebeca eval

[x] append adapter-processed semantic embeds to set pre-cached prompts (better initialization)
[] condition on 
  [x] scores
  - can likely improve design here
  [] text prompt
  - ensure text embeds come last so 0th first embed is always providing target score
  [] demographics

[] val should show inputted images + their scores
[] qual eval
  [] pad & drop so that we could give just e.g. 2 images instead of four
[] quant eval
