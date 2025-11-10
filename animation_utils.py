import bioviz


def bioviz_animate(biorbd_model_path_with_mesh, q, result_folder, name):
    b = bioviz.Viz(
        biorbd_model_path_with_mesh,
        mesh_opacity=1.0,
        background_color=(1, 1, 1),
        show_local_ref_frame=False,
        show_markers=False,
        show_segments_center_of_mass=False,
        show_global_center_of_mass=False,
        show_global_ref_frame=False,
        show_gravity_vector=False,
        show_floor=False,
    )
    b.set_camera_zoom(0.39)
    b.maximize()
    b.update()
    b.load_movement(q)

    b.start_recording(f"videos/{result_folder}/" + name + ".ogv")
    for frame in range(q.shape[1] + 1):
        b.movement_slider[0].setValue(frame)
        b.add_frame()
    b.stop_recording()

    b = bioviz.Kinogram(model_path=biorbd_model_path_with_mesh,
                        mesh_opacity=1.0,
                        background_color=(1, 1, 1),
                        show_local_ref_frame=False,
                        show_markers=False,
                        show_segments_center_of_mass=False,
                        show_global_center_of_mass=False,
                        show_global_ref_frame=False,
                        show_gravity_vector=False,
                        show_floor=False,
                        )
    b.load_movement(q)
    b.set_camera_zoom(0.39)
    b.maximize()
    b.update()
    b.exec(frame_step=2,
           save_path=f"videos/{result_folder}/kinograms/{name}.png")

