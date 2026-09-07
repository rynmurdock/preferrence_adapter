
Note: currently need to patch and then run within updated training

[] integrate preference prior + gaze-conditioned diffusion's relevant code
  [] update training +
  [] prior adapter architecture
[] re-enable demographics conditioning

[] integrate Gabriel's rebeca eval

[] append adapter-processed semantic embeds to set pre-cached prompts (better initialization)

[] condition on 
  [] scores
  - can likely improve design here
  [] text prompt
  - ensure text embeds come last so 0th first embed is always providing target score
  [] demographics

[] val should show inputted images + their scores

[] qual eval

[] quant eval
