# Launch server

```
python inference_server.py --checkpoint_path $CKP_PATH --decoder_cfg decoder_kp_cfg.yml
```

# Launch client

```
python inference_client.py --host $HOST --image_path $IMAGES_FOLDER
```
