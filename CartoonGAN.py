from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class CartoonGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'CartoonGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.init_epoch = args.init_epoch # args.epoch // 20
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.vgg_weight = args.vgg_weight
        self.ld = args.ld

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainB_smooth_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB_smooth'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# the number of discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, use_bias=False, scope='conv_s2_'+str(i))
                x = conv(x, channel*2, kernel=3, stride=1, pad=1, use_bias=False, scope='conv_s1_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            # Bottleneck
            for i in range(self.n_res):
                x = resblock(x, channel, use_bias=False, scope='resblock_' + str(i))

            # Up-Sampling
            for i in range(2) :
                x = deconv(x, channel//2, kernel=3, stride=2, use_bias=False, scope='deconv_'+str(i))
                x = conv(x, channel//2, kernel=3, stride=1, pad=1, use_bias=False, scope='up_conv_'+str(i))
                x = instance_norm(x, scope='up_ins_norm_'+str(i))
                x = relu(x)

                channel = channel // 2


            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, scope='G_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        channel = self.ch // 2
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='conv_s2_' + str(i))
                x = lrelu(x, 0.2)

                x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='conv_s1_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='last_conv')
            x = instance_norm(x, scope='last_ins_norm')
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='D_logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type.__contains__('dragan') :
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit = self.discriminator(interpolated, reuse=True, scope=scope)


        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        GP = 0
        # WGAN - LP
        if self.gan_type.__contains__('lp'):
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan' :
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))


        return GP

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')


        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
        trainB_smooth = tf.data.Dataset.from_tensor_slices(self.trainB_smooth_dataset)

        gpu_device = '/gpu:0'
        trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB_smooth = trainB_smooth.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()
        trainB_smooth_iterator = trainB_smooth.make_one_shot_iterator()


        self.real_A = trainA_iterator.get_next()
        self.real_B = trainB_iterator.get_next()
        self.real_B_smooth = trainB_smooth_iterator.get_next()

        self.test_real_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_real_A')


        """ Define Generator, Discriminator """
        self.fake_B = self.generator(self.real_A)

        real_B_logit = self.discriminator(self.real_B)
        fake_B_logit = self.discriminator(self.fake_B, reuse=True)
        real_B_smooth_logit = self.discriminator(self.real_B_smooth, reuse=True)


        """ Define Loss """
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan') :
            GP = self.gradient_panalty(real=self.real_B, fake=self.fake_B) + self.gradient_panalty(self.real_B, fake=self.real_B_smooth)
        else :
            GP = 0.0

        v_loss = self.vgg_weight * vgg_loss(self.real_A, self.fake_B)
        g_loss = self.adv_weight * generator_loss(self.gan_type, fake_B_logit)
        d_loss = self.adv_weight * discriminator_loss(self.gan_type, real_B_logit, fake_B_logit, real_B_smooth_logit) + GP

        self.Vgg_loss = v_loss
        self.Generator_loss = g_loss + v_loss
        self.Discriminator_loss = d_loss


        """ Result Image """
        self.test_fake_B = self.generator(self.test_real_A, reuse=True)

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.init_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Vgg_loss, var_list=G_vars)
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)


        """" Summary """
        self.G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_gan = tf.summary.scalar("G_gan", g_loss)
        self.G_vgg = tf.summary.scalar("G_vgg", v_loss)

        self.V_loss_merge = tf.summary.merge([self.G_vgg])
        self.G_loss_merge = tf.summary.merge([self.G_loss, self.G_gan, self.G_vgg])
        self.D_loss_merge = tf.summary.merge([self.D_loss])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            # lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            if self.decay_flag :
                lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)

            for idx in range(start_batch_id, self.iteration):

                train_feed_dict = {
                    self.lr : lr
                }

                if epoch < self.init_epoch :
                    # Init G
                    real_A_images, fake_B_images, _, v_loss, summary_str = self.sess.run([self.real_A, self.fake_B,
                                                                             self.init_optim,
                                                                             self.Vgg_loss, self.V_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%3d] [%5d/%5d] time: %4.4f v_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, v_loss))

                else :
                    # Update D
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    # Update G
                    g_loss = None
                    if (counter - 1) % self.n_critic == 0 :
                        real_A_images, fake_B_images, _, g_loss, summary_str = self.sess.run([self.real_A, self.fake_B,
                                                                                              self.G_optim,
                                                                                              self.Generator_loss, self.G_loss_merge], feed_dict = train_feed_dict)
                        self.writer.add_summary(summary_str, counter)
                        past_g_loss = g_loss

                    if g_loss == None:
                        g_loss = past_g_loss
                    print("Epoch: [%3d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # display training status
                counter += 1


                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(real_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B_images, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)



            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                         self.gan_type, n_res, n_dis,
                                                         self.n_critic, self.sn,
                                                         int(self.adv_weight), int(self.vgg_weight))

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_real_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        index.close()
