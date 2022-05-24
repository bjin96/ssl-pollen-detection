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
            decay: float
    ):
        super().__init__()
        self.decay = decay

        self.student = student
        self.teacher = teacher

    @no_grad()
    def update_teacher(self):
        for teacher_parameter, student_parameter in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_parameter.data.copy_(self.decay * teacher_parameter + (1 - self.decay) * student_parameter)

        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            teacher_buffer.data.copy_(self.decay * teacher_buffer + (1 - self.decay) * student_buffer)
