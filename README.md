## Binary Dice Loss and Dice Coefficient

**How to use**

- Binary Dice Loss

  ```python
  from binaryDice_Loss_Score import binaryDiceLoss
  
  in train...
  
  criterion = binaryDiceLoss
  loss = criterion(img_batch, ground_truth_mask_batch)
  epoch_loss += loss.item()
  
  optimizer.zero_grad()  
  loss.backward()  
  optimizer.step()  
  
  ```

  

- Binary Dice Coefficient

  ```python
  from binaryDice_Loss_Score import binaryDiceCoeff
  
  # case of 1 mask & ground truth
  sum_dice = 0
  dice_score = binaryDiceCoeff(predicted_mask, ground_truth_mask)
  sum_dice += dice_score
  
  result_dice = sum_dice/total_number_of_images
  
  # case of batch of mask & ground truth
  sum_dice = 0
  dice_score = binaryDiceCoeff(predicted_mask_batch, ground_truth_mask_batch)
  sum_dice += dice_score
  
  result_dice = sum_dice/total_number_of_images
  
  # you can evaluate total images using 'for' sentence
  ```

  

