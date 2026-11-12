# Project Plan

## Demo
A web app on a laptop at the poster.
1. **Picking a photo.** Visitior uploads a photo from their phone or picks one from a gallery of choices
2. **Play against the model (optional).** Visitor clicks on the world map where they think the photo was taken. Models guess appears after, along with the true location and distances away from the true location in km.
3. **What gave it away?** A toggle overlays the detected clues on the photo and a second toggle shows heatmap of where the model looked
4. **Privacy protection.** One button blurs the photo using the privacy model. Visitor can see the original and the protected photo side by side, the model's new guess, how far the guess moved and an evaluation of image quality.

Practical details:
- The gallery of photo choices is of precomputed examples. It is used a fallback if live processing is slow.
- Uploaded photos are not store and GPS metadata is stripped on upload

## Main experiments
Test sets:
- Im2GPS (237 everyday Flickr photos), for comparison with PlaNet.
- OSV-5M test subset (street-level photos).

### Experiment 1 - Geolocation Accuracy

### Experiment 2 - Detector Accuracy

### Experiment 3 - Privacy Accuracy

## Minimum result
Abdu:
- Frozen ViT + final layer trained

William:
- One-layer decoder trained and evaluated

Victoria:
- Optimized blur and trained privacy network working

Group:
- Demo with precomputed gallery

## Stretch Goal
- User can choose parts of the picture to blur before a model guesses
- Leaderboard which shows who makes the best blurring

# Owners
