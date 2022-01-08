import torch
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

        self.student_parameters = list(student.parameters())
        self.teacher_parameters = list(teacher.parameters())

    @no_grad()
    def update_teacher(self):
        for teacher_parameter, student_parameter in zip(self.teacher_parameters, self.student_parameters):
            teacher_parameter.data.copy_(self.decay * teacher_parameter + (1 - self.decay) * student_parameter)
