all_array = list(range(1,101))
batch_size = 10
tr_epochs = 10

start_batch = 0
end_batch = batch_size

total_batch = int(100/batch_size)

for epoch in range(tr_epochs):
    for i in range(total_batch):
        print "Start of batch", i
        batch_x = all_array[start_batch:end_batch]

        for item in batch_x:
            print item

        start_batch += batch_size
        end_batch += batch_size
        print "End of batch", i
    start_batch = 0
    end_batch = batch_size
    print "End of Epoch", epoch
