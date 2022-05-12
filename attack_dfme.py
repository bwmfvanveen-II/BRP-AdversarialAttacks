from attack import *
from grad_approx import estimate_gradient
from nets import NetGenMnist


class DFMEAttack(MSAttack):

    def __init__(self, args, defense_obj=None):
        super(DFMEAttack, self).__init__(args, defense_obj)

        self.z_dim = args.z_dim

        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

    def set_netG(self):
        if self.args.dataset == 'MNIST':
            self.netG = NetGenMnist(z_dim=self.z_dim).to(self.device)

    def dfme_train_netS(self, path_s, path_g=None):
        """
        Training the substitute net using DFME.
        """
        print("Starting training net S using \'DFME\'")
        self.set_netG()

        optimizer_s = torch.optim.SGD(self.netS.parameters(), lr=self.args.lr_tune_s, momentum=0.9)
        optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr_tune_g)

        steps = sorted([int(step * self.args.epoch_dg) for step in self.args.steps])
        print("Learning rate scheduling at steps: ", steps)
        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, steps, self.args.scale)
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, steps, self.args.scale)

        ce_criterion = nn.CrossEntropyLoss()

        torch.save(self.netS.state_dict(), path_s + "_start.pth")
        torch.save(self.netG.state_dict(), path_g + "_start.pth")

        for epoch in range(self.args.epoch_dg):
            print("***********************")
            print("global epoch: %d/%d" % (epoch + 1, self.args.epoch_dg))
            print('global epoch {0} lr_s: {1} lr_g: {2}'.format(epoch + 1, optimizer_s.param_groups[0]['lr'], optimizer_g.param_groups[0]['lr']))

            for e_iter in range(self.args.epoch_itrs):
                for ng in range(self.args.epoch_dg_g):
                    self.netG.zero_grad()

                    with torch.no_grad():
                        if self.args.cuda:
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).cuda()
                        else:
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).cpu()

                    x_query = self.netG(noise)
                    approx_grad_wrt_x, loss_g, q_num = estimate_gradient(self.args, self.msd.netV, self.netS, x_query, loss_type="l1")

                    x_query.backward(approx_grad_wrt_x)

                    optimizer_g.step()

                    if e_iter == self.args.epoch_itrs - 1:
                        print("loss_g:", loss_g.cpu().detach().numpy())

                for ns in range(self.args.epoch_dg_s):
                    self.netS.zero_grad()

                    with torch.no_grad():
                        if self.args.cuda:
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).cuda()
                        else:
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).cpu()

                    x_query = self.netG(noise).detach()
                    s_output = self.netS(x_query)
                    # s_output_p = F.softmax(s_output, dim=1)

                    with torch.no_grad():
                        v_output = self.msd.netV(x_query)
                        v_output_p = F.softmax(v_output, dim=1)

                    v_logit = v_output
                    v_logit = F.log_softmax(v_logit, dim=1).detach()
                    v_logit -= v_logit.mean(dim=1).view(-1, 1).detach()
                    loss_s = F.l1_loss(s_output, v_logit)

                    loss_s.backward()
                    optimizer_s.step()

                    if (e_iter == self.args.epoch_itrs - 1) and (ns == self.args.epoch_dg_s - 1):
                        print("ns idx:", ns, "loss_s:", loss_s.cpu().detach().numpy())

            scheduler_s.step()
            scheduler_g.step()

            # save results in lists
            acc = comm.accuracy(self.netS, 'netS', test_loader=self.test_loader, cuda=self.args.cuda)

        # save the final model
        torch.save(self.netS.state_dict(), path_s + "_over.pth")
        torch.save(self.netG.state_dict(), path_g + "_over.pth")
        print("Finished training of netS")

    def ce_loss(self, q, p):
        return torch.mean(-torch.mean(p * torch.log(q+1e-8), dim=1))