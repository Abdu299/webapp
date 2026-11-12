hyper_params = {
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "num_workers": 8,
}

osv5m = {
    "name": "osv5m",  # the judge, also the output folder
    "batch_size": 16,
    "area_weight": 3.0,  # cost of blurring the whole photo
    "smoothness_weight": 1.0,  # cost of a noisy mask
    "floor": 0.01,  # probability near the truth below which a hidden photo gets no gradient
}

# short run to check speed and that val/hidden and val/area move
pilot = osv5m | {"limit_train_batches": 64}

# area weight sweep, compared on validation
area1 = osv5m | {"area_weight": 1.0}
area10 = osv5m | {"area_weight": 10.0}
area30 = osv5m | {"area_weight": 30.0}
area2 = osv5m | {"area_weight": 2.0}
area7 = osv5m | {"area_weight": 7.0}
area15 = osv5m | {"area_weight": 15.0}

# smoothness weight sweep
tv3 = osv5m | {"smoothness_weight": 3.0}
tv10 = osv5m | {"smoothness_weight": 10.0}
tv20 = osv5m | {"smoothness_weight": 20.0}

# one value per 7x7 pixels (32x32) like optimised blur
block4 = osv5m | {"block": 4}
block7 = osv5m | {"block": 7}
block14 = osv5m | {"block": 14}

# Abdu's GeoViT as judge and encoder, same loss weights as osv5m
geovit = osv5m | {"name": "geovit"}

geovit_area1 = geovit | {"area_weight": 1.0}
geovit_area2 = geovit | {"area_weight": 2.0}
geovit_area7 = geovit | {"area_weight": 7.0}
geovit_area10 = geovit | {"area_weight": 10.0}
geovit_area15 = geovit | {"area_weight": 15.0}
geovit_area30 = geovit | {"area_weight": 30.0}
geovit_tv3 = geovit | {"smoothness_weight": 3.0}
geovit_tv10 = geovit | {"smoothness_weight": 10.0}
geovit_tv20 = geovit | {"smoothness_weight": 20.0}
geovit_block4 = geovit | {"block": 4}
geovit_block7 = geovit | {"block": 7}
geovit_block14 = geovit | {"block": 14}
