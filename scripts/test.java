for (int attempt = 0; attempt < numAttempts; attempt++) {
    int bestCol = 0;
    for (int i = 0; i < img.width-attempt; i++) {
      if (seamFitness[bestCol][img.height-1] > seamFitness[i][img.height-1]) {
        bestCol = i;
      }
    }
  
    for (int y = newImg.height-1; y >= 0; y--) {
      boolean pastBestCol = false;
  
      for (int x = 0; x < newImg.width; x++) {
        if (x == bestCol) {
          pastBestCol = true;
        }
  
        int newLoc = x + y*newImg.width;
        int oldLoc = (pastBestCol ? x+1 : x) + y*img.width;
  
        newImg.pixels[newLoc] = img.pixels[oldLoc];
      }
  
      if (y > 0) {
        // update best column for the next row
        float theMin = seamFitness[bestCol][y-1];
  
        if (bestCol > 0 && seamFitness[bestCol-1][y-1] <= theMin) {
          bestCol = bestCol - 1;
        } else if (bestCol < img.width-1 && seamFitness[bestCol+1][y-1] <= theMin) {
          bestCol = bestCol + 1;
        }
      }
    }
  }
