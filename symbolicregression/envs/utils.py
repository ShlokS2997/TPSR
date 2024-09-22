# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico


def unsqueeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = unsqueeze_dic(dico[d])
        else:
            dico_copy[d] = [dico[d]]
    return dico_copy


def squeeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = squeeze_dic(dico[d])
        else:
            dico_copy[d] = dico[d][0]
    return dico_copy


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def getSizeOfNestedList(listOfElem):
    """Get number of elements in a nested list"""
    count = 0
    for elem in listOfElem:
        if isinstance(elem, list):
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count


class ZMQNotReady(Exception):
    pass


class ZMQNotReadySample:
    pass


def load_data(env):
    data = np.load('your_data_file.npy')  # Adjust based on your data source
    return data


def create_data_loaders(training_data, validation_data):
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)  # Adjust batch size as needed
    valid_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    return train_loader, valid_loader


def reset_parameters(env):
    env.model.initialize_parameters()  # Replace with your actual model parameter reset method


def train_one_epoch(env, train_loader):
    for batch in train_loader:
        # Forward pass
        outputs = env.model(batch['inputs'])
        loss = compute_loss(outputs, batch['targets'])  # Define compute_loss based on your loss function
        # Backward pass
        loss.backward()
        env.optimizer.step()  # Update weights
        env.optimizer.zero_grad()  # Reset gradients


def evaluate(env, valid_loader):
    total_loss = 0
    for batch in valid_loader:
        with torch.no_grad():
            outputs = env.model(batch['inputs'])
            loss = compute_loss(outputs, batch['targets'])
            total_loss += loss.item()
    avg_loss = total_loss / len(valid_loader)
    logger.info(f'Validation Loss: {avg_loss}')
