from torch import no_grad
from torch.nn import Module


class ExponentialMovingAverage(Module):
    """
    Exponential moving average (EMA) calculation for a teacher-student model.
    """

    def __init__(
            self,
            student: Module,
            teacher: Module,
            ramp_up_decay: float = 0.99,
            after_ramp_up_decay: float = 0.999,
            ramp_up_epochs: float = 3,
    ):
        super().__init__()
        self.decay = ramp_up_decay

        self.ramp_up_epochs = ramp_up_epochs
        self.after_ramp_up_decay = after_ramp_up_decay

        self.student = student
        self.teacher = teacher

    @no_grad()
    def update_teacher(self, epoch):
        if epoch == self.ramp_up_epochs:
            self.decay = self.after_ramp_up_decay

        for teacher_parameter, student_parameter in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_parameter.data.copy_(self.decay * teacher_parameter + (1 - self.decay) * student_parameter)

        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            teacher_buffer.data.copy_(self.decay * teacher_buffer + (1 - self.decay) * student_buffer)
