"""Neural network model definitions used across the federated learning experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN1(torch.nn.Module):
    def __init__(self, h1=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 28*28))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits for CrossEntropyLoss
        return x
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        # FIX: Flatten the image inside the model so it works with ALL your algorithms!
        x = x.view(x.size(0), -1)

        features = self.relu(self.fc1(x))
        logits = self.fc2(features)

        if return_features:
            return logits, features
        return logits
class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            # 1. Pointwise Convolution (Expansion)
            layers.append(nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            # FIX: Switched to LeakyReLU to prevent "dying" neurons
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        layers.extend([
            # 2. Depthwise Convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # FIX: Switched to LeakyReLU
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # 3. Pointwise Convolution (Projection)
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class SimpleNN_CIFAR100(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        # Standard MobileNetV2 expansion is usually 6
        # We use 6 here for better accuracy
        exp = 6

        self.features = nn.Sequential(
            # Input: 32x32
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # Blocks
            # 32x32 -> 32x32
            InvertedResidual(in_c=32, out_c=32, stride=1, expand_ratio=1),

            # 32x32 -> 16x16
            InvertedResidual(in_c=32, out_c=64, stride=2, expand_ratio=exp),
            InvertedResidual(in_c=64, out_c=64, stride=1, expand_ratio=exp),

            # 16x16 -> 8x8
            InvertedResidual(in_c=64, out_c=128, stride=2, expand_ratio=exp),
            InvertedResidual(in_c=128, out_c=128, stride=1, expand_ratio=exp),
            InvertedResidual(in_c=128, out_c=128, stride=1, expand_ratio=exp), # Added one extra block for depth

            # 8x8 -> 8x8 (Final width expansion)
            InvertedResidual(in_c=128, out_c=256, stride=1, expand_ratio=exp),

            # NEW: Final Projection Layer (Crucial for Accuracy)
            # Expands 256 -> 1280 to give the classifier rich features
            nn.Conv2d(256, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            # Now we classify from a vector of 1280, not 256
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def _initialize_weights(self):
        # Same Kaiming init as before
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG16_CIFAR101(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10, self).__init__()

        # Configuration for VGG16:
        # Numbers are filters, 'M' is MaxPool
        self.cfg = [
            64, 64, 'M',                   # Block 1
            128, 128, 'M',                 # Block 2
            256, 256, 256, 'M',            # Block 3
            512, 512, 512, 'M',            # Block 4
            512, 512, 512, 'M'             # Block 5
        ]

        self.features = self._make_layers(self.cfg)

        # Classifier
        # Because we pooled 5 times on 32x32 input, the spatial size is 1x1.
        # Input channels = 512.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512), # BN often helps in the classifier too for VGG
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) # Flatten
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x), # Essential for VGG16 depth
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.3)
                ]
                in_channels = x

        return nn.Sequential(*layers)


class VGG16_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10, self).__init__()

        # Configuration for VGG16:
        # Numbers are filters, 'M' is MaxPool
        self.cfg = [
            64, 64, 'M',                   # Block 1
            128, 128, 'M',                 # Block 2
            256, 256, 256, 'M',            # Block 3
            512, 512, 512, 'M',            # Block 4
            512, 512, 512, 'M'             # Block 5
        ]

        self.features = self._make_layers(self.cfg)

        # Classifier
        # Because we pooled 5 times on 32x32 input, the spatial size is 1x1.
        # Input channels = 512.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512), # BN often helps in the classifier too for VGG
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # 1. ADD 'return_features=False' TO THE SIGNATURE
    def forward(self, x, return_features=False):
        out = self.features(x)
        out = out.view(out.size(0), -1) # Flatten

        # 2. CAPTURE THE FEATURES BEFORE THE CLASSIFIER
        features = out

        # 3. PASS THROUGH THE CLASSIFIER
        out = self.classifier(out)

        # 4. RETURN BOTH IF REQUESTED
        if return_features:
            return out, features

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x), # Essential for VGG16 depth
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.3)
                ]
                in_channels = x

        return nn.Sequential(*layers)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)

        # nn.Linear automatically applies to the last dimension.
        # out shape becomes: (batch_size, seq_length, vocab_size)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_size,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # FIXED: Changed hidden_dim to hidden_size
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden

class CNN_FEMNIST(nn.Module):
    """Standard lightweight CNN for 28x28 grayscale FEMNIST (62 classes)"""
    def __init__(self):
        super(CNN_FEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62) # 62 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
