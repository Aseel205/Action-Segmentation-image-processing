#from extract_frames import extract_frames
#extract_frames("videos/Video_1.mp4", "frames/demo", fps=3)


#from extract_features import extract_features_from_folder
#extract_features_from_folder("frames/demo", "features/demo.npy")



from prepare_labels import save_frame_labels

save_frame_labels("annotations.txt", "features/demo.npy", "labels/demo_labels.npy", fps=3)
