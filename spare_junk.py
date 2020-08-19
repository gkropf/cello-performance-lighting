

# Generate expected auto-correlation for all notes
autocorr_funcs = pd.DataFrame(columns=[str(k) for k in range(len(x2))])
for k in range(len(all_notes)):
	row = all_notes.iloc[k]
	if wave_form == 'sine':
		autocorr_funcs.loc[row['note']] = autocorr(sin(row['freq']*(x1/1000)*2*pi))
	elif wave_form == 'square':
		autocorr_funcs.loc[row['note']] = autocorr(signal.square(row['freq']*(x1/1000)*2*pi))
	elif wave_form == 'saw':
		autocorr_funcs.loc[row['note']] = autocorr(signal.sawtooth(row['freq']*(x1/1000)*2*pi))


		# Determine primary frequency and note using comparison method
		# comp_func = lambda x: corrcoef(x,signal_autocorr)[0,1]
		# temp = autocorr_funcs.apply(comp_func, axis=1)
		# curr_note = temp.index[argmax(temp.values)]
		# text2.set_text(f'Main Note ({curr_note})')
		# line3.set_ydata(autocorr_funcs.loc[curr_note].values)