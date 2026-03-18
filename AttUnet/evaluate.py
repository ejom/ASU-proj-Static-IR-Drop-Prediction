model.load_state_dict(torch.load('/content/drive/MyDrive/saved/ft_real/499.pth'))
model.eval()

dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
dataloader_test_original_size = torch.utils.data.DataLoader(dataset=dataset_test_original_size, batch_size=1, shuffle=False)

l1_sum = 0
f1_sum = 0
for i, (data, data_org) in enumerate(zip(dataloader_test, dataloader_test_original_size)):
    maps = data[:, :-1, :, :].to(device)
    ir = data_org[:, -1, :, :].unsqueeze(1)
    shape = ir.shape
    with torch.no_grad():
        output, x = model(maps)
    output = output / 100
    output = output.cpu().numpy()
    output = torch.tensor(resize(output, shape))

    l1 = L1(output, ir).item()
    f1 = F1_Score(output.numpy().copy(), ir.numpy().copy())[0]
    l1_sum += l1
    f1_sum += f1
    print(f'Test {i}: L1={l1:.6f}, F1={f1:.4f}')

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
    sns.heatmap(output.numpy()[0, 0, :], ax=axs[0])
    axs[0].set_title('Predicted')
    sns.heatmap(ir.numpy()[0, 0, :], ax=axs[1])
    axs[1].set_title('Ground Truth')
    plt.suptitle(f'Test Case {i}')
    plt.show()

print(f'\nAvg L1: {l1_sum/10:.8f}, Avg F1: {f1_sum/10:.4f}')