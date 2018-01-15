import data_utils
from tensorflow.python.platform import gfile
import tensorflow as tf
import pac_model
import math
import statistics
import numpy as np
import pickle
import os
import re
import datetime
import time

def initialize_params_helper(params, 
                                stage_of_development,
                                experiment_dir_name,
                                experiment_dir_suffix,
                                params_initialization_for_training=None,
                                params_initialization_for_resume_training=None,
                                params_initialization_for_evaluation=None):

    directories_to_create_experiment_folders_in = ['logs']
    filenames_of_images_zero_labeled = None
    labels_of_images_zero_labeled = None
    filenames_of_images_one_labeled = None
    labels_of_images_one_labeled = None
    filenames_of_images = []
    labels_of_images = []

    if stage_of_development == 'training':
        for key_parameter in params_initialization_for_training:
            params[key_parameter] = params_initialization_for_training[key_parameter]
        params['experiment_dir'] = data_utils.create_experiment_folders(experiment_dir_suffix, directories_to_create_experiment_folders_in)
        if params['experiment_dir'] == None:
            return None, None, None
        filenames_of_images_zero_labeled, labels_of_images_zero_labeled = data_utils.prepare_data(params['training_path_zero_labeled'])
        filenames_of_images_one_labeled, labels_of_images_one_labeled = data_utils.prepare_data(params['training_path_one_labeled'])
        file_to_save_pickle = params['logs_dir'] + "/" + params['experiment_dir'] + "/" + "parameters.pickle"
        pickle.dump(params, open(file_to_save_pickle, "wb"))
    else:
        directories_to_search_in = ['logs']
        experiment_directory_name = data_utils.find_experiment_name(experiment_dir_name, experiment_dir_suffix, directories_to_search_in)
        if experiment_directory_name == None:
            return None, None, None
        if experiment_directory_name == None:
            print("There was an error in identifying the experiment directory to evaluate model.")
            print("Please select an experiment that has been trained with a model.")
            return
        if params['logs_dir'] == None:
            print("Please provide the path to the logs directory that stores pickle files of saved parameters of different experiment runs.")
            return
        params['experiment_dir'] = experiment_directory_name
        file_to_load_pickle = params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle"
        if not os.path.isfile(file_to_load_pickle):
            print(params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle")
            print("Parameters pickel file does not exist.")
            print("Please select an experiment that has been trained with a model.")
            return
        params = pickle.load(open(file_to_load_pickle, "rb"))
        if stage_of_development == 'resume_training':
            params['resume_training'] = True
            for key_parameter in params_initialization_for_resume_training:
                params[key_parameter] = params_initialization_for_resume_training[key_parameter]
            filenames_of_images_zero_labeled, labels_of_images_zero_labeled = data_utils.prepare_data(params['training_path_zero_labeled'])
            filenames_of_images_one_labeled, labels_of_images_one_labeled = data_utils.prepare_data(params['training_path_one_labeled'])
        else:
            params['forward_only'] = True
            params['evaluate_model'] = True
            for key_parameter in params_initialization_for_evaluation:
                params[key_parameter] = params_initialization_for_evaluation[key_parameter]
            filenames_of_images_zero_labeled, labels_of_images_zero_labeled = data_utils.prepare_data(params['dev_path_zero_labeled'])
            filenames_of_images_one_labeled, labels_of_images_one_labeled = data_utils.prepare_data(params['dev_path_one_labeled'])
    longer_list_filenames = None
    longer_list_labels = None
    shorter_list_filenames = None
    shorter_list_labels = None
    if len(filenames_of_images_zero_labeled) < len(filenames_of_images_one_labeled):
        longer_list_filenames = filenames_of_images_one_labeled
        longer_list_labels = labels_of_images_one_labeled
        shorter_list_filenames = filenames_of_images_zero_labeled
        shorter_list_labels = labels_of_images_zero_labeled
    else:
        shorter_list_filenames = filenames_of_images_one_labeled
        shorter_list_labels = labels_of_images_one_labeled
        longer_list_filenames = filenames_of_images_zero_labeled
        longer_list_labels = labels_of_images_zero_labeled
    if len(shorter_list_filenames) == 0:
        filenames_of_images = longer_list_filenames
        labels_of_images = longer_list_labels
    else:
        for i in range(0, len(longer_list_filenames)):
            filenames_of_images.append(longer_list_filenames[i])
            filenames_of_images.append(shorter_list_filenames[i % len(shorter_list_filenames)])
            labels_of_images.append(longer_list_labels[i])
            labels_of_images.append(shorter_list_labels[i % len(shorter_list_labels)])
    return params, filenames_of_images, labels_of_images

def initialize_params(experiment_dir_name,
                        experiment_dir_suffix,
                        stage_of_development,
                        params_initialization_for_training=None,
                        params_initialization_for_resume_training=None,
                        params_initialization_for_evaluation=None):
    params = {}
    params['experiment_dir'] = None
    params['resume_training'] = False
    params['evaluate_model'] = False
    params['resume_training'] = False
    params['num_epochs'] = None
    params['max_steps'] = None
    params['num_steps_before_checkpoint'] = 1
    params['data_dir'] = "data"
    params['logs_dir'] = "logs"
    params['training_path'] = None
    params['dev_path'] = None
    params['testing_path'] = None
    params['forward_only'] = False

    params, filenames_of_images, labels_of_images = initialize_params_helper(params,
                                                                                stage_of_development,
                                                                                experiment_dir_name,
                                                                                experiment_dir_suffix,
                                                                                params_initialization_for_training=params_initialization_for_training,
                                                                                params_initialization_for_resume_training=params_initialization_for_resume_training,
                                                                                params_initialization_for_evaluation=params_initialization_for_evaluation)
    return params, filenames_of_images, labels_of_images

def create_model(session, params):
    model = pac_model.HygeineDetectionModel(session, params['batch_size'], params['stage_of_development'], params['learning_rate'], params['learning_rate_decay_factor'])
    ckpt = tf.train.get_checkpoint_state(params['logs_dir'] + "/" + params['experiment_dir'])
    if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and 
                (params['resume_training'] or params['evaluate_model'])):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        session.run(tf.tables_initializer())
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        session.run(tf.tables_initializer())
    return model

def run_training_step_with_feed_dictionary(params,
                                            sess,
                                            model,
                                            batched_filenames,
                                            batched_labels,
                                            step_time,
                                            loss,
                                            current_step,
                                            previous_losses,
                                            options=None,
                                            run_metadata=None):
    start_time_for_step = time.time()
    step_loss = model.step_with_dictionary(sess, batched_filenames, batched_labels)
    loss += (step_loss / params['num_steps_before_checkpoint'])
    step_time += (time.time() - start_time_for_step) / params['num_steps_before_checkpoint']
    current_step += 1
    # Once in a while, we save checkpoint, print statistics, and run evals.
    if current_step % params['num_steps_before_checkpoint'] == 0:
        with tf.device("/cpu:0"):
            # Print statistics for the previous epoch.
            print ("global step %d learning rate %.4f step-time %.2f loss " 
                    "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]): 
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path_directory = params['logs_dir'] + "/" + params['experiment_dir']
            checkpoint_path = os.path.join(checkpoint_path_directory, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
    return step_time, loss, current_step, previous_losses

def run_evaluation_step_for_predictions_with_feed_dictionary(params,
                                                                sess,
                                                                model,
                                                                batched_filenames,
                                                                batched_labels,
                                                                current_step):
    predictions = model.step_with_dictionary(sess, batched_filenames, batched_labels)
    num_of_correct_predictions = 0
    numpy_predictions = np.array(predictions).astype(np.int32)
    numpy_real_labels = np.array(batched_labels).astype(np.int32)
    num_of_correct_predictions += (numpy_predictions==numpy_real_labels).sum()
    print("Accuracy For Current Batch: %s " % ((num_of_correct_predictions * 1.0) / (len(predictions) * 1.0)))
    return (num_of_correct_predictions * 1.0) / (len(predictions) * 1.0)

def run_training_with_feed_dictionary(params, gpu_device, filenames_of_training_images, labels_of_training_images):
    with tf.Graph().as_default(), tf.device(gpu_device):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            if len(filenames_of_training_images) < params['batch_size']:
                params['batch_size'] = len(filenames_of_training_images)
            model = create_model(sess, params)
            start_of_preprocessing = time.time()
            preprocessed_training_images = data_utils.preprocess_images(filenames_of_training_images)
            end_of_preprocessing = time.time()
            print("Length of Time (in seconds) for preprocessing %s" % (end_of_preprocessing - start_of_preprocessing))
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            for num_epochs in range(0, params['num_epochs']):
                max_num_steps_for_given_epoch = len(preprocessed_training_images) // params['batch_size']
                if params['max_steps'] != None:
                    if params['max_steps'] <= max_num_steps_for_given_epoch:
                        total_num_steps_for_given_epoch = params['max_steps']
                    else:
                        total_num_steps_for_given_epoch = max_num_steps_for_given_epoch
                else:
                    total_num_steps_for_given_epoch = max_num_steps_for_given_epoch
                for timestep in range(0, total_num_steps_for_given_epoch):
                    current_batch_of_preprocessed_training_images = preprocessed_training_images[timestep * params['batch_size']: (timestep+1)*params['batch_size']]
                    current_batch_of_labels_of_training_images = labels_of_training_images[timestep * params['batch_size']: (timestep+1)*params['batch_size']]
                    print("Batch Size of List of Filenames %s" % len(current_batch_of_preprocessed_training_images))
                    print("Batch Size of List of Label Images %s" % len(current_batch_of_labels_of_training_images))
                    step_time, loss, current_step, previous_losses = run_training_step_with_feed_dictionary(params,
                                                                                                            sess,
                                                                                                            model,
                                                                                                            current_batch_of_preprocessed_training_images,
                                                                                                            current_batch_of_labels_of_training_images,
                                                                                                            step_time,
                                                                                                            loss,
                                                                                                            current_step,
                                                                                                            previous_losses)
                if params['max_steps'] == None or params['max_steps'] > max_num_steps_for_given_epoch:
                    if max_num_steps_for_given_epoch * params['batch_size'] < len(preprocessed_training_images):                                                                                           
                        last_batch_of_preprocessed_training_images = preprocessed_training_images[(max_num_steps_for_given_epoch * params['batch_size']):]
                        leftover = params['batch_size'] - (len(preprocessed_training_images) - (max_num_steps_for_given_epoch * params['batch_size']))
                        last_batch_of_preprocessed_training_images.extend(preprocessed_training_images[:leftover])

                        last_batch_of_labels_of_training_images = labels_of_training_images[(max_num_steps_for_given_epoch * params['batch_size']):]
                        leftover = params['batch_size'] - (len(labels_of_training_images) - (max_num_steps_for_given_epoch * params['batch_size']))
                        last_batch_of_labels_of_training_images.extend(labels_of_training_images[:leftover])

                        print("Batch Size of List of LAST Filenames %s" % len(last_batch_of_preprocessed_training_images))
                        print("Batch Size of List of LAST Label Images %s" % len(last_batch_of_labels_of_training_images))
                        step_time_loss, loss, current_step, previous_losses = run_training_step_with_feed_dictionary(params,
                                                                                                                sess,
                                                                                                                model,
                                                                                                                last_batch_of_preprocessed_training_images,
                                                                                                                last_batch_of_labels_of_training_images,
                                                                                                                step_time,
                                                                                                                loss,
                                                                                                                current_step,
                                                                                                                previous_losses)

def evaluate_model_with_feed_dictionary(params, gpu_device, filenames_of_evaluation_images, labels_of_evaluation_images):
    with tf.Graph().as_default(), tf.device(gpu_device):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            # Create model.
            if len(filenames_of_evaluation_images) < params['batch_size']:
                params['batch_size'] = len(filenames_of_evaluation_images)
            model = create_model(sess, params)
            print("Done creating the model.")
            start_of_preprocessing = time.time()
            preprocessed_evaluation_images = data_utils.preprocess_images(filenames_of_evaluation_images)
            end_of_preprocessing = time.time()
            print("Length of Time (in seconds) for preprocessing %s" % (end_of_preprocessing - start_of_preprocessing))
            current_step = 0
            step_time_for_checkpoint = 0.0
            list_of_accuracies = []
            for num_epochs in range(0, params['num_epochs']):
                max_num_steps_for_given_epoch = len(preprocessed_evaluation_images) // params['batch_size']
                if params['max_steps'] != None:
                    if params['max_steps'] <= max_num_steps_for_given_epoch:
                        total_num_steps_for_given_epoch = params['max_steps']
                    else:
                        total_num_steps_for_given_epoch = max_num_steps_for_given_epoch
                else:
                    total_num_steps_for_given_epoch = max_num_steps_for_given_epoch
                for timestep in range(0, total_num_steps_for_given_epoch):
                    current_step += 1
                    res_from_evaluation_step = None
                    current_batch_of_preprocessed_evaluation_images = preprocessed_evaluation_images[timestep * params['batch_size']: (timestep+1)*params['batch_size']]
                    current_batch_of_labels_of_evaluation_images = labels_of_evaluation_images[timestep * params['batch_size']: (timestep+1)*params['batch_size']]
                    start_time_for_step = time.time()
                    accuracy_from_evaluation_step = run_evaluation_step_for_predictions_with_feed_dictionary(params,
                                                                                                                sess,
                                                                                                                model,
                                                                                                                current_batch_of_preprocessed_evaluation_images,
                                                                                                                current_batch_of_labels_of_evaluation_images,
                                                                                                                current_step)
                    list_of_accuracies.append(accuracy_from_evaluation_step)
                if params['max_steps'] == None or params['max_steps'] > max_num_steps_for_given_epoch:
                    if max_num_steps_for_given_epoch * params['batch_size'] < len(preprocessed_evaluation_images):
                        current_step += 1                                                                                          
                        last_batch_of_preprocessed_evaluation_images = preprocessed_evaluation_images[(max_num_steps_for_given_epoch * params['batch_size']):]
                        leftover = params['batch_size'] - (len(preprocessed_evaluation_images) - (max_num_steps_for_given_epoch * params['batch_size']))
                        last_batch_of_preprocessed_evaluation_images.extend(preprocessed_evaluation_images[:leftover])

                        last_batch_of_labels_of_evaluation_images = labels_of_evaluation_images[(max_num_steps_for_given_epoch * params['batch_size']):]
                        leftover = params['batch_size'] - (len(labels_of_evaluation_images) - (max_num_steps_for_given_epoch * params['batch_size']))
                        last_batch_of_labels_of_evaluation_images.extend(labels_of_evaluation_images[:leftover])
                        accuracy_from_evaluation_step = run_evaluation_step_for_predictions_with_feed_dictionary(params,
                                                                                                                    sess,
                                                                                                                    model,
                                                                                                                    last_batch_of_preprocessed_evaluation_images,
                                                                                                                    last_batch_of_labels_of_evaluation_images,
                                                                                                                    current_step)
                    list_of_accuracies.append(accuracy_from_evaluation_step)
            if len(list_of_accuracies) > 0:
                print("Average Accuracy Across Batches: %s " % statistics.mean(list_of_accuracies))
            else:
                print("Not enough examples to evaluate model. Please have at least 1 example.")


