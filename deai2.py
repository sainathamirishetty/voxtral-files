def save_minutes():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".docx",
        filetypes=[("Word Document", "*.docx"), ("All files", "*.*")]
    )

    if file_path:
        try:
            # Retrieve all meeting details
            meeting_date = widgets['date_entry'].get()
            meeting_time = widgets['time_entry'].get()
            meeting_venue = widgets['venue_entry'].get()
            meeting_agenda = widgets['agenda_entry'].get()
            additional_context = widgets['context_text'].get("1.0", tk.END).strip()

            save_minutes_text = live_minutes_widget['minutes_display'].get("1.0", tk.END)

            # Compose content to save
            content = (
                f"Meeting Date: {meeting_date}\n"
                f"Meeting Time: {meeting_time}\n"
                f"Venue: {meeting_venue}\n"
                f"Agenda: {meeting_agenda}\n\n"
                f"Additional Context:\n{additional_context}\n\n"
                f"Minutes:\n{save_minutes_text}"
            )

            # --- DOCX Save (formatted) ---
            try:
                from html2docx import html2docx
                import markdown2

                save_minutes_text_html = markdown2.markdown(content)
                with open(file_path, "wb") as fp:
                    doc_content = html2docx(save_minutes_text_html, title="Minutes")
                    fp.write(doc_content.getvalue())

                update_status(f"Minutes saved to {file_path}")

            except Exception as docx_err:
                update_status(f"Error saving DOCX: {str(docx_err)}")

            # --- Plain TXT fallback ---
            txt_file = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_file, "w", encoding="utf-8") as txt_fp:
                txt_fp.write(content)

            update_status(f"Plain text backup saved to {txt_file}")

        except Exception as e:
            update_status(f"Error saving minutes: {str(e)}")
